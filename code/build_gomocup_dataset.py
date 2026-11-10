"""Convert Gomocup .psq game records into our training dataset.

Gomocup runs a freestyle 15x15 league -- the exact rules this project plays -- and
its engines (RAPFI, KATAGOMO, ALPHAGOMOKU, ...) are far stronger than anything in
this repository. That makes these records the best label source available: the
winner of each game is known, so every position in it gets an exact outcome label
with no rollouts and no sampling noise.

The one weakness is opening variety: only 12 openings are used per league, so the
first few plies repeat across thousands of games. The midgame and endgame are
diverse.

.psq format (Piskvork). The first line is a header whose `x:y` field is only the
standard first-move marker, NOT a move -- the body holds the complete game:

    Piskvorky 15x15, 11:11, 0
    12,14,0                        <- x,y,milliseconds
    ...
    STARPOINT.zip                  <- White
    RAPFI25.zip                    <- Black
    -1
    1,Freestyle15_1

Coordinates are 1-indexed, column then row, counted from the top-left.

    python code/build_gomocup_dataset.py --out gomocup15.npz
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import paths
from dataset import pack_boards
from engine import CELLS, W, has_five

LEAGUES = ("Freestyle15_1", "Freestyle15_2")


def parse(path):
    """-> list of (x, y) 1-indexed placements, in play order."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    moves = []
    for ln in lines[1:]:
        bits = ln.split(",")
        if len(bits) < 2:
            break
        try:
            moves.append((int(bits[0]), int(bits[1])))
        except ValueError:
            break
    return moves


def replay(moves):
    """-> (winner colour, number of plies played) or None if illegal/undecided."""
    bb = [0, 0]
    occ = 0
    for i, (x, y) in enumerate(moves):
        idx = (y - 1) * W + (x - 1)
        if not (0 <= idx < CELLS) or (occ >> idx) & 1:
            return None
        c = i % 2
        bb[c] |= 1 << idx
        occ |= 1 << idx
        if has_five(bb[c]):
            return c, i + 1
    return None


def onehot(cls_arr):
    """(N,) class ids -> (N,3) float32 with column order [loss, draw, win]."""
    out = np.zeros((len(cls_arr), 3), dtype=np.float32)
    out[np.arange(len(cls_arr)), cls_arr] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gomocup15.npz")
    ap.add_argument("--root", default=os.path.join("data", "gomocup"))
    ap.add_argument("--leagues", default=",".join(LEAGUES))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.path.dirname(args.out):
        args.out = paths.data(args.out)

    rng = np.random.default_rng(args.seed)
    owns, opps, lasts, plies, gids, clss = [], [], [], [], [], []
    move_out, vt_out, vw_out = [], [], []
    games = kept = skipped = 0
    ply_hist = {}

    for league in [s.strip() for s in args.leagues.split(",") if s.strip()]:
        files = sorted(glob.glob(os.path.join(args.root, league, "*.psq")))
        for path in files:
            games += 1
            moves = parse(path)
            r = replay(moves)
            if r is None:
                skipped += 1
                continue
            winner, n = r
            kept += 1
            gid = games
            bb = [0, 0]
            occ = 0
            last = -1
            for k in range(1, n):
                idx = (moves[k - 1][1] - 1) * W + (moves[k - 1][0] - 1)
                bb[(k - 1) % 2] |= 1 << idx
                occ |= 1 << idx
                last = idx
                mover = k % 2
                cls = 2 if winner == mover else 0
                owns.append(bb[mover])
                opps.append(bb[mover ^ 1])
                lasts.append(last)
                plies.append(k)
                gids.append(gid)
                clss.append(cls)
                # policy target: the move the engine actually played here
                move_out.append((moves[k][1] - 1) * W + (moves[k][0] - 1))
                vt_out.append(1.0 if cls == 2 else -1.0)
                vw_out.append(1.0)
                ply_hist[k] = ply_hist.get(k, 0) + 1
        print(f"  {league}: {len(files)} files", flush=True)

    print(f"\ngames     : {games}")
    print(f"kept      : {kept}  ({kept/max(games,1):.1%})")
    print(f"skipped   : {skipped}  (illegal, or no five-in-a-row)")
    print(f"positions : {len(clss):,}")

    cls = np.asarray(clss, dtype=np.uint8)
    ply = np.asarray(plies, dtype=np.uint8)
    print(f"  win  (+1): {(cls == 2).sum():,}  ({(cls == 2).mean():.1%})")
    print(f"  loss (-1): {(cls == 0).sum():,}  ({(cls == 0).mean():.1%})")

    gid = np.asarray(gids, dtype=np.int64)
    uniq = np.unique(gid)
    val_games = set(rng.choice(uniq, size=max(1, int(len(uniq) * args.val_frac)),
                               replace=False).tolist())
    is_val = np.isin(gid, list(val_games))

    meta = {
        "source": "Gomocup 2026 freestyle 15x15 league (.psq records)",
        "ruleset": "freestyle, 15x15, no forbidden moves, overline wins",
        "labels": "exact game outcome from the mover's perspective",
        "caveat": "only 12 openings per league, so early plies repeat",
        "games": kept,
        "positions": int(len(cls)),
        "split": "by game",
    }
    import json
    np.savez_compressed(
        args.out,
        own=pack_boards(owns),
        opp=pack_boards(opps),
        last=np.asarray(lasts, dtype=np.int16),
        ply=ply,
        cls=cls,
        game_id=gid,
        is_val=is_val,
        move=np.asarray(move_out, dtype=np.int16),
        value_target=onehot(np.asarray(clss, dtype=np.uint8)),
        value_weight=np.asarray(vw_out, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )
    print(f"\nsaved -> {args.out}  "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"validation: {len(val_games)} games, {is_val.sum():,} positions")


if __name__ == "__main__":
    main()
