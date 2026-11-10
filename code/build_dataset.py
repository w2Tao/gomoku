"""Build solver-labelled value data for 15x15 freestyle gomoku.

Pipeline
--------
    random self-play game
        -> every position in it
        -> VCF(depth) proves +1  -> keep, class WIN
           else every legal move loses -> keep, class LOSS
           else -> drop (unknown; never feed 'unknown' to the loss)
        -> store canonically (own / opp / last), no colour bookkeeping

The output is deliberately compact: bitboards are split into two uint64 words,
so a record is ~40 bytes. Board planes and the 8-fold dihedral augmentation are
built at load time by `dataset.py`, not stored.

Splits are assigned per *game*, never per position: consecutive plies of the
same game differ by one stone and would leak.
"""
from __future__ import annotations

import paths

import argparse
import json
import random
import time

import numpy as np

from dataset import pack_boards
from engine import CELLS, has_five
from solvers import _near_stones
from vcf import vcf


def play_random_game(rng):
    bb = [0, 0]
    empt = list(range(CELLS))
    turn = 0
    last = -1
    hist = []
    while empt:
        hist.append((bb[0], bb[1], turn, last))
        k = rng.randrange(len(empt))
        cell = empt[k]
        empt[k] = empt[-1]
        empt.pop()
        bb[turn] |= 1 << cell
        last = cell
        if has_five(bb[turn]):
            return hist, turn
        turn ^= 1
    return hist, None


def label(att, dfn, depth):
    """Return 2 (win) / 0 (loss) / None (unknown)."""
    if vcf(att, dfn, depth):
        return 2
    occ = att | dfn
    cand = _near_stones(occ)
    if not cand:
        return None
    for m in cand:
        a2 = att | (1 << m)
        if has_five(a2):
            return None
        if not vcf(dfn, a2, depth):
            return None
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50000,
                    help="stop once this many labelled positions are collected")
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--out", default=paths.data("batch1.npz"))
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    args = ap.parse_args()
    if not os.path.dirname(args.out):      # bare name -> land in data/
        args.out = paths.data(args.out)

    rng = random.Random(args.seed)

    owns, opps = [], []
    lasts, clss, plies, gids = [], [], [], []

    t0 = time.time()
    game_id = 0
    games = 0
    raw = 0
    n_win = n_loss = 0

    while len(clss) < args.target and time.time() - t0 < args.max_seconds:
        hist, winner = play_random_game(rng)
        gid = game_id
        game_id += 1
        games += 1
        for ply, (bb0, bb1, turn, last) in enumerate(hist):
            raw += 1
            cls = label(bb0 if turn == 0 else bb1, bb1 if turn == 0 else bb0,
                        args.depth)
            if cls is None:
                continue
            own = bb0 if turn == 0 else bb1
            opp = bb1 if turn == 0 else bb0
            owns.append(own)
            opps.append(opp)
            lasts.append(last)
            clss.append(cls)
            plies.append(ply)
            gids.append(gid)
            if cls == 2:
                n_win += 1
            else:
                n_loss += 1
        if games % 100 == 0:
            dt = time.time() - t0
            print(f"games {games:5d}  raw {raw:7d}  labelled {len(clss):6d}  "
                  f"({raw/dt:6.0f} raw/s, {len(clss)/dt:6.0f} lab/s)", flush=True)
            if len(clss) >= args.target:
                break

    n = len(clss)
    dt = time.time() - t0
    if n == 0:
        raise SystemExit("no labelled positions were produced")

    gids_arr = np.asarray(gids, dtype=np.uint32)
    split = (gids_arr % args.val_every == 0).astype(np.uint8)

    out = {
        "own": pack_boards(owns),
        "opp": pack_boards(opps),
        "last": np.asarray(lasts, dtype=np.int16),
        "cls": np.asarray(clss, dtype=np.uint8),
        "ply": np.asarray(plies, dtype=np.uint8),
        "game_id": gids_arr,
        "is_val": split,
        "meta": np.array(json.dumps({
            "board": "15x15 freestyle gomoku",
            "value": "optimal-play win expectancy from the side to move",
            "depth": args.depth,
            "seed": args.seed,
            "games": games,
            "raw_positions": raw,
            "labelled": n,
            "win": n_win,
            "loss": n_loss,
            "seconds": round(dt, 1),
        })),
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)

    print()
    print(f"labelled positions : {n}  from {raw} raw, {games} games, {dt:.0f}s")
    val_games = len(set(gids[i] for i in range(n) if split[i]))
    print(f"win  (+1)          : {n_win:6d}  {n_win/n:.1%}")
    print(f"loss (-1)          : {n_loss:6d}  {n_loss/n:.1%}")
    print(f"win:loss ratio     : {n_win/max(n_loss,1):.2f} : 1")
    print(f"validation         : {int(split.sum())} positions "
          f"from {val_games} held-out games")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
