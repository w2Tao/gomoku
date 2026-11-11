"""Distil a champion engine's search into training labels.

The Gomocup corpus gives one label per position: the move some engine played, from
games of mixed strength. RAPFI narrates its search on MESSAGE lines, which carries
two richer signals:

  Eval <n>        a continuous score for the position it searched
  the PV          a whole line, so one call labels many positions

The PV alternates sides. Starting from RAPFI's own turn, PV[0] is its move, PV[1]
is the reply it expects from us -- and that odd-indexed half is exactly the signal
the policy head needs: "in a position where I am to move, this is what the champion
would play". PV[0] alone is just the move we already get from TURN.

Our model supplies the opening randomness: engine-vs-engine games are deterministic
from an empty board (no reliable way to hand an engine a position), so every game
would be identical.

    python code/build_distill_dataset.py --games 150 --time-ms 300 --out dist1.npz
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

import baselines
import paths
from dataset import pack_boards
from engine import CELLS, has_five
from fetch_engines import ENGINE
from play import PolicyAgent
from pbrain import EngineError, Pbrain, find_exe, rapfi_targets

OPENING = 2


def onehot(cls_arr):
    out = np.zeros((len(cls_arr), 3), dtype=np.float32)
    out[np.arange(len(cls_arr)), cls_arr] = 1.0
    return out


def walk_pv(b0, b1, last, pv, mover):
    """Label every position along the PV, in canonical mover form.

    `mover` is whose turn it is at the head of the line. Returns
    [(own, opp, last, cell, mover, depth_in_line, eval_known)].
    """
    rows = []
    bb = [b0, b1]
    prev = last
    turn = mover
    for k, cell in enumerate(pv):
        if (bb[0] | bb[1]) >> cell & 1:
            break                      # PV left the game we know; stop
        own, opp = (bb[0], bb[1]) if turn == 0 else (bb[1], bb[0])
        rows.append((own, opp, prev, cell, turn, k, k == 0))
        bb[turn] |= 1 << cell
        prev = cell
        if has_five(bb[turn]):
            break
        turn ^= 1
    return rows


def play_game(eng, agent, rng, timeout, engine_black, opening=OPENING):
    """RAPFI (one colour) vs our model (the other) -> harvested rows.

    `opening` random plies lead in. A wider opening pushes the teacher into a much
    wider spread of positions, which matters because the PVs are conditioned on the
    opponent: with a narrow opening the teacher only ever shows how to beat our own
    model, and that transfers poorly to a real engine.
    """
    our_colour = 1 if engine_black else 0
    engine_colour = 1 - our_colour
    bb = [0, 0]
    last = -1
    rows = []
    random_left = opening

    mv = eng.begin(engine_black)
    turn = 0

    for _ in range(CELLS):
        occ = bb[0] | bb[1]
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return rows

        if turn == engine_colour:
            if mv is None:
                raise EngineError("engine owed a move but sent nothing")
            x, y = mv
            cell = y * 15 + x
            if not (0 <= cell < CELLS) or (occ >> cell) & 1:
                raise EngineError(f"engine returned an illegal cell {(x, y)}")
            _eval, pv = rapfi_targets(eng.last_messages)
            if pv and pv[0] != cell:
                pv = [cell] + [c for c in pv if c != cell]   # trust TURN over the PV
            if len(pv) > 1:
                rows.extend(walk_pv(bb[0], bb[1], last, pv, turn))
            else:
                own, opp = (bb[0], bb[1]) if turn == 0 else (bb[1], bb[0])
                rows.append((own, opp, last, cell, turn, 0, True))
        elif random_left > 0:
            cell = rng.choice(legal)
            random_left -= 1
        else:
            cell = agent.move(bb[turn], bb[turn ^ 1], last, rng)
            if cell is None:
                return rows

        bb[turn] |= 1 << cell
        last = cell
        if has_five(bb[turn]):
            return rows
        turn ^= 1
        if turn == engine_colour:
            mv = eng.play(last % 15, last // 15, timeout)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="RAPFI25")
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--time-ms", type=int, default=300)
    ap.add_argument("--out", default="dist1.npz")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0,
                    help="which slice of --games this process plays")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--opening", type=int, default=OPENING,
                    help="random plies before the teacher takes over; "
                         "wider spreads the positions it is asked about")
    args = ap.parse_args()
    out = args.out if os.path.dirname(args.out) else paths.data(args.out)

    meta = ENGINE().get(args.engine, {})
    exe = find_exe(os.path.join("data", "engines", args.engine),
                   meta.get("prefer"))

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, man = baselines.load_baseline(args.model, dev)
    agent = PolicyAgent(net, dev)
    rng = random.Random(args.seed + 1000 * args.shard)
    # each shard plays its own slice: separate seed, and the colour alternation is
    # offset per shard so shards do not replay the same openings
    per_shard = (args.games + args.shards - 1) // args.shards

    eng = Pbrain(exe, time_ms=args.time_ms, stall_s=meta.get("stall_s", 10.0),
                 begin_white=meta.get("begin_white", False))
    rows, done, errors = [], 0, 0
    for g in range(1, per_shard + 1):
        engine_black = (g + args.shard) % 2 == 0
        try:
            eng.start()
            got = play_game(eng, agent, rng, args.time_ms * 10,
                           engine_black, args.opening)
        except EngineError:
            errors += 1
            continue
        finally:
            try:
                eng.close()
            except Exception:
                pass
        rows.extend(got)
        done += 1
        if g % 25 == 0:
            print(f"  {g} games  {len(rows):,} labels  "
                  f"({errors} aborted)", flush=True)

    if not rows:
        sys.exit("no labels")
    ours = sum(1 for r in rows if r[4] != 0)     # positions where we are to move
    print(f"\ngames {done}  aborted {errors}  labels {len(rows):,}")

    # the mover won whenever the position is inside the teacher's winning line;
    # we do not know the result, so value labels come only from the head position
    # of each line and are left at "unknown" elsewhere
    clss = np.full(len(rows), 1, dtype=np.uint8)          # 1 = draw/unknown
    gid = np.zeros(len(rows), dtype=np.int64)
    block = max(1, len(rows) // max(done, 1))
    for i in range(len(rows)):
        gid[i] = i // block

    rng2 = np.random.default_rng(args.seed)
    games = np.unique(gid)
    val = set(rng2.choice(games, size=max(1, int(len(games) * 0.1)),
                          replace=False).tolist())

    data = dict(
        own=pack_boards([r[0] for r in rows]),
        opp=pack_boards([r[1] for r in rows]),
        last=np.asarray([r[2] for r in rows], dtype=np.int16),
        ply=np.zeros(len(rows), dtype=np.uint8),
        cls=clss,
        game_id=gid,
        is_val=np.isin(gid, list(val)),
        move=np.asarray([r[3] for r in rows], dtype=np.int16),
        value_target=onehot(clss),
        value_weight=np.zeros(len(rows), dtype=np.float32),
        meta=np.array(json.dumps({
            "source": f"{args.engine} PV distillation at {args.time_ms} ms",
            "labels": len(rows),
            "our_side_positions": int(ours),
            "note": "policy labels only; value targets are unset",
        })),
    )
    np.savez_compressed(out, **data)
    print(f"our-side positions {ours:,} | saved -> {out} "
          f"({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
