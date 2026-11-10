"""Generate a teacher dataset from the frozen heuristic playing itself.

Schema-identical to what `build_gomocup_dataset.py` produces, so the same trainer
consumes both and the only thing that differs between experiments is the label
source:

    own, opp, last, ply, game_id, is_val, cls, move, value_target, value_weight

* policy target  = the move the teacher would play in this position
* value target   = whether the side to move went on to win that game

That mirrors the Gomocup records exactly (engine move, engine game outcome); the
only difference is how strong the mover is.

Sharded across subprocesses because the heuristic is pure Python and CPU bound.
`multiprocessing` cannot be used here: its queues are named pipes, which the
sandbox blocks.

    python code/build_teacher_dataset.py --positions 150000 --workers 12
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np

import paths
from dataset import pack_boards
from engine import CELLS, W, has_five
from play import frozen_heuristic

OPENING = 4          # random plies before the teacher takes over


def play_one(rng, teacher):
    """-> (moves, winner colour) for one self-play game. Random opening first."""
    bb = [0, 0]
    last = -1
    moves = []
    for ply in range(OPENING):
        occ = bb[0] | bb[1]
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        m = rng.choice(legal)
        bb[ply % 2] |= 1 << m
        moves.append(m)
        last = m
        if has_five(bb[ply % 2]):
            return moves, ply % 2
    turn = OPENING % 2
    for _ in range(CELLS):
        occ = bb[0] | bb[1]
        if occ == (1 << CELLS) - 1:
            return moves, None
        m = teacher.move(bb[turn], bb[turn ^ 1], last, rng)
        if m is None:
            return moves, None
        bb[turn] |= 1 << m
        moves.append(m)
        last = m
        if has_five(bb[turn]):
            return moves, turn
        turn ^= 1
    return moves, None


def rows_from_game(moves, winner, gid):
    """Yield (own, opp, last, ply, gid, cls, move, value) per decided position."""
    if winner is None:
        return
    bb = [0, 0]
    occ = 0
    out = []
    for k, m in enumerate(moves):
        if k < OPENING:
            # the opening is random, not the teacher's opinion
            bb[k % 2] |= 1 << m
            occ |= 1 << m
            continue
        mover = k % 2
        cls = 2 if winner == mover else 0
        out.append((bb[mover], bb[mover ^ 1], moves[k - 1] if k else -1,
                    k, gid, cls, m, 1.0 if cls == 2 else -1.0))
        bb[k % 2] |= 1 << m
        occ |= 1 << m
    return out


def build(target, seed, shard, shards):
    rng = random.Random(seed * 1000 + shard)
    teacher = frozen_heuristic()
    rows = []
    gid = 0
    while len(rows) < target:
        moves, winner = play_one(rng, teacher)
        gid += 1
        rows.extend(rows_from_game(moves, winner, gid) or [])
        if gid % 100 == 0:
            print(f"  shard {shard}: {gid} games, {len(rows)} positions",
                  flush=True)
    return rows[:target]


def pack(rows, seed, val_frac):
    owns = [r[0] for r in rows]
    opps = [r[1] for r in rows]
    lasts = [r[2] for r in rows]
    plies = [r[3] for r in rows]
    gids = [r[4] for r in rows]
    clss = [r[5] for r in rows]
    mvs = [r[6] for r in rows]
    vals = [r[7] for r in rows]
    gid = np.asarray(gids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    uniq = np.unique(gid)
    val_games = set(rng.choice(uniq, size=max(1, int(len(uniq) * val_frac)),
                               replace=False).tolist())
    return dict(
        own=pack_boards(owns),
        opp=pack_boards(opps),
        last=np.asarray(lasts, dtype=np.int16),
        ply=np.asarray(plies, dtype=np.uint8),
        cls=np.asarray(clss, dtype=np.uint8),
        game_id=gid,
        is_val=np.isin(gid, list(val_games)),
        move=np.asarray(mvs, dtype=np.int16),
        value_target=onehot(np.asarray(clss, dtype=np.uint8)),
        value_weight=np.ones(len(rows), dtype=np.float32),
    )


def onehot(cls_arr):
    """(N,) class ids -> (N,3) float32 with column order [loss, draw, win]."""
    out = np.zeros((len(cls_arr), 3), dtype=np.float32)
    out[np.arange(len(cls_arr)), cls_arr] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=150000)
    ap.add_argument("--out", default="teacher15.npz")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--shards", type=int, default=0)
    args = ap.parse_args()

    # --workers is the user-facing knob; it maps onto shard count
    if args.shards == 0:
        args.shards = max(1, args.workers)

    out = args.out if os.path.dirname(args.out) else paths.data(args.out)

    if args.shard < 0 and args.shards > 1:
        per = max(1, args.positions // args.shards)
        parts = []
        procs = []
        for i in range(args.shards):
            part = os.path.join(paths.DATA, f"_teacher_part{i}.npz")
            parts.append(part)
            procs.append(subprocess.Popen([
                sys.executable, os.path.abspath(__file__),
                "--positions", str(per), "--shard", str(i),
                "--shards", str(args.shards), "--seed", str(args.seed),
                "--out", part, "--val-frac", str(args.val_frac),
            ]))
        for p in procs:
            p.wait()
        merged = {}
        offset = 0
        for part in parts:
            with np.load(part) as z:
                for k in z.files:
                    merged.setdefault(k, []).append(z[k])
            os.remove(part)
        off = 0
        cat = {}
        for k, chunks in merged.items():
            if k == "meta":
                continue                      # scalar string, rebuilt below
            if k == "game_id":
                fixed = []
                for c in chunks:
                    fixed.append(c + off)
                    off += int(c.max()) + 1
                cat[k] = np.concatenate(fixed)
            else:
                cat[k] = np.concatenate(chunks)
        cat["meta"] = np.array(json.dumps({
            "source": "frozen heuristic self-play",
            "teacher": "play.FROZEN_HEURISTIC",
            "labels": "teacher move + game outcome from the mover's perspective",
            "positions": int(len(cat["cls"])),
            "split": "by game",
        }))
        np.savez_compressed(out, **cat)
        print(f"\nsaved -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
        print(f"positions: {len(cat['cls']):,}")
        return

    rows = build(args.positions, args.seed, max(args.shard, 0), args.shards)
    data = pack(rows, args.seed, args.val_frac)
    meta = {
        "source": "frozen heuristic self-play",
        "teacher": "play.FROZEN_HEURISTIC",
        "labels": "teacher move + game outcome from the mover's perspective",
        "positions": int(len(rows)),
        "split": "by game",
    }
    data["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(out, **data)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
