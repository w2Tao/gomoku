"""Build Monte-Carlo value labels for the gap the solver leaves.

Why this file exists
--------------------
The solver-labelled set contains only positions with V* = +-1, because a proof
only exists at the extremes. A network trained on it learns a hard binary
pattern detector and cannot rank moves -- measured: 90% of arbitrary positions
get |V| > 0.5, and the trained net's playing strength is identical to a
randomly-initialised one.

So: for every position the solver *cannot* judge, estimate V by sampling:

    V_MC(s) = (#rollouts the mover wins) / N  -  (#rollouts the mover loses) / N

Rollouts are uniform-random continuations. That is a weak policy, but the
estimate is *graded*, and grading is exactly what the saturation problem lacks.

Solver-decidable positions are skipped on purpose: labelling them with a soft
~0.6 would contradict the exact +-1 label for the same kind of position, and the
model would learn to hedge where it should be certain.
"""
from __future__ import annotations

import paths

import argparse
import json
import os
import random
import time

import numpy as np

from build_dataset import label, play_random_game
from dataset import pack_boards
from engine import CELLS, playout


def rollout_counts(bb0, bb1, turn, n, rng, empt):
    w = d = 0
    for _ in range(n):
        r = playout(bb0, bb1, turn, rng, empt)
        if r is None:
            d += 1
        elif r == turn:
            w += 1
    return w, d, n - w - d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=25000)
    ap.add_argument("--rollouts", type=int, default=128)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--sample-rate", type=float, default=0.35)
    ap.add_argument("--out", default=paths.data("mc1.npz"))
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--val-every", type=int, default=10)
    args = ap.parse_args()
    if not os.path.dirname(args.out):      # bare name -> land in data/
        args.out = paths.data(args.out)

    rng = random.Random(args.seed)

    owns, opps = [], []
    lasts, plies, gids = [], [], []
    nw, nd, nl = [], [], []

    t0 = time.time()
    gid = 0
    games = 0
    raw = skipped = 0

    while len(gids) < args.target:
        hist, winner = play_random_game(rng)
        occ = hist[-1][0] | hist[-1][1] if hist else 0
        this_gid = gid
        gid += 1
        games += 1
        for ply, (bb0, bb1, turn, last) in enumerate(hist):
            if rng.random() > args.sample_rate:
                continue
            raw += 1
            att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
            if label(att, dfn, args.depth) is not None:
                skipped += 1
                continue
            empt = [i for i in range(CELLS) if not (((bb0 | bb1) >> i) & 1)]
            w, d, l = rollout_counts(bb0, bb1, turn, args.rollouts, rng, empt)
            own, opp = att, dfn
            owns.append(own)
            opps.append(opp)
            lasts.append(last)
            plies.append(ply)
            gids.append(this_gid)
            nw.append(w)
            nd.append(d)
            nl.append(l)
        if games % 100 == 0:
            dt = time.time() - t0
            print(f"games {games:5d}  sampled {raw:7d}  kept {len(gids):6d}  "
                  f"({len(gids)/dt:6.0f} kept/s)", flush=True)

    n = len(gids)
    dt = time.time() - t0
    gids_arr = np.asarray(gids, dtype=np.uint32)
    # column order MUST match the class convention used everywhere else:
    # 0 = loss, 1 = draw, 2 = win
    r = np.stack([np.asarray(nl), np.asarray(nd), np.asarray(nw)],
                 1).astype(np.uint16)
    p = r.astype(np.float32) / r.sum(1, keepdims=True)
    v = p[:, 2] - p[:, 0]          # P(win) - P(loss)

    out = {
        "own": pack_boards(owns),
        "opp": pack_boards(opps),
        "last": np.asarray(lasts, dtype=np.int16),
        "ply": np.asarray(plies, dtype=np.uint8),
        "game_id": gids_arr,
        "counts": r,
        "is_val": (gids_arr % args.val_every == 0).astype(np.uint8),
        "meta": np.array(json.dumps({
            "source": "monte-carlo rollouts",
            "rollouts": args.rollouts,
            "solver_depth": args.depth,
            "solver_decidable_skipped": skipped,
            "games": games,
            "positions": n,
            "mean_V": float(v.mean()),
            "std_V": float(v.std()),
            "seconds": round(dt, 1),
        })),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)

    print()
    print(f"positions          : {n:,}  from {raw:,} sampled, {skipped:,} "
          f"skipped (solver-decidable), {games:,} games, {dt:.0f}s")
    print(f"rollouts/position  : {args.rollouts}")
    print(f"V_MC mean / std    : {v.mean():+.3f} / {v.std():.3f}")
    print(f"share |V_MC|>0.5   : {(np.abs(v) > 0.5).mean():.1%}   "
          f"(solver set: 100%)")
    print(f"validation         : {int(out['is_val'].sum()):,} positions, "
          f"{len(set(g for g, s in zip(gids, out['is_val']) if s))} games")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
