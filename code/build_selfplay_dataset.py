"""Generate self-play data from a baseline, optionally merged with engine records.

Supervised imitation of engine moves has plateaued: more data, more capacity and
value-guided search all failed to beat the baseline beyond noise. What those
experiments never changed is the *distribution* -- the model only ever saw
positions strong engines reached. Self-play sees the positions the model itself
reaches, which is where its own mistakes live.

    python code/build_selfplay_dataset.py --model baseline --games 400 --out sp1.npz

Output schema matches `build_gomocup_dataset.py` exactly, so `train_policy.py`
consumes either. With `--base`, the two are concatenated (game ids kept disjoint),
which is the recommended first iteration: pure self-play from a weak policy tends
to collapse, and mixing keeps the engine signal in the pool.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

import baselines
import paths
from dataset import load, pack_boards
from engine import CELLS, has_five
from play import PolicyAgent

OPENING = 4          # random plies before the model takes over


def onehot(cls_arr):
    out = np.zeros((len(cls_arr), 3), dtype=np.float32)
    out[np.arange(len(cls_arr)), cls_arr] = 1.0
    return out


def play_one(agent, rng):
    """-> list of (own, opp, last, cell) plus the winner.

    Each entry is the position *before* the move, with `last` being the previous
    (opponent) stone. Recording after the move puts the label on an occupied cell,
    which the policy loss masks to -inf and blows up.
    """
    bb = [0, 0]
    last = -1
    seq = []
    for ply in range(OPENING):
        occ = bb[0] | bb[1]
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        m = rng.choice(legal)
        seq.append((bb[0], bb[1], last, m))
        bb[ply % 2] |= 1 << m
        last = m
        if has_five(bb[ply % 2]):
            return seq, ply % 2
    turn = OPENING % 2
    for _ in range(CELLS):
        occ = bb[0] | bb[1]
        if occ == (1 << CELLS) - 1:
            return seq, None
        m = agent.move(bb[turn], bb[turn ^ 1], last, rng)
        if m is None:
            return seq, None
        seq.append((bb[0], bb[1], last, m))
        bb[turn] |= 1 << m
        last = m
        if has_five(bb[turn]):
            return seq, turn
        turn ^= 1
    return seq, None


def rows_from_game(seq, winner, gid):
    if winner is None:
        return []
    out = []
    for ply, (b0, b1, prev, m) in enumerate(seq):
        if ply < OPENING:
            continue                      # random, not the model's opinion
        mover = ply % 2
        own, opp = (b0, b1) if mover == 0 else (b1, b0)
        cls = 2 if winner == mover else 0
        out.append((own, opp, prev, ply, gid, cls, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--out", default="sp1.npz")
    ap.add_argument("--base", default=None,
                    help="existing dataset to concatenate, e.g. gomocup15.npz")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out = args.out if os.path.dirname(args.out) else paths.data(args.out)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, man = baselines.load_baseline(args.model, dev)
    agent = PolicyAgent(net, dev)
    print(f"self-play with {args.model} ({man['parameters']:,} params) on {dev}")

    rng = random.Random(args.seed)
    rows = []
    for g in range(1, args.games + 1):
        seq, winner = play_one(agent, rng)
        rows.extend(rows_from_game(seq, winner, g))
        if g % 100 == 0:
            print(f"  {g} games, {len(rows):,} positions", flush=True)

    if not rows:
        sys.exit("no usable games (all drawn?)")

    owns = [r[0] for r in rows]
    opps = [r[1] for r in rows]
    clss = [r[5] for r in rows]
    gid = np.asarray([r[4] for r in rows], dtype=np.int64)
    plies = np.asarray([r[3] for r in rows], dtype=np.uint8)

    data = dict(
        own=pack_boards(owns),
        opp=pack_boards(opps),
        last=np.asarray([r[2] for r in rows], dtype=np.int16),
        ply=plies,
        cls=np.asarray(clss, dtype=np.uint8),
        game_id=gid,
        move=np.asarray([r[6] for r in rows], dtype=np.int16),
        value_target=onehot(np.asarray(clss, dtype=np.uint8)),
        value_weight=np.ones(len(rows), dtype=np.float32),
    )
    print(f"\nself-play: {args.games} games, {len(rows):,} positions, "
          f"win {(np.asarray(clss) == 2).mean():.1%}")

    if args.base:
        base = load(paths.data(args.base))
        n_base = len(base["cls"])
        # offset must clear the BASE's ids, not the self-play ones, or merged
        # game ids collide and the by-game train/val split leaks
        offset = int(base["game_id"].max()) + 1
        merged = {}
        for k in ("own", "opp", "last", "ply", "cls", "game_id", "move",
                  "value_target", "value_weight"):
            a, b = base[k], data[k]
            if k == "game_id":
                b = b + offset
            merged[k] = np.concatenate([a, b])
        n = len(merged["cls"])
        data = merged
        print(f"merged with {args.base}: {n_base:,} + {len(rows):,} = {n:,} positions")

    # split by game, whichever path produced the data
    rng2 = np.random.default_rng(args.seed)
    games = np.unique(data["game_id"])
    val = set(rng2.choice(games, size=max(1, int(len(games) * 0.1)),
                          replace=False).tolist())
    data["is_val"] = np.isin(data["game_id"], list(val))
    data["meta"] = np.array(json.dumps({
        "source": f"self-play of {args.model}"
                  + (f" merged with {args.base}" if args.base else ""),
        "positions": int(len(data["cls"])),
        "split": "by game",
    }))
    np.savez_compressed(out, **data)
    print(f"saved -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
