"""Experiment F: does depth actually buy coverage?

exp_e showed depth 3 -> 5 -> 7 gaining almost nothing. This checks whether that
is real saturation or an artefact, on a much larger sample, and also measures the
-negative- side, which is where depth should matter most.
"""

import random
import sys
import time

from engine import CELLS, has_five
from labeler import proven_win
from solvers import _near_stones
from vcf import forcing_moves, vcf

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
DEPTHS = [3, 5, 7, 9, 11]

rng = random.Random(20261110)


def sample(n):
    pos = []
    while len(pos) < n:
        b = [0, 0]
        empt = list(range(CELLS))
        turn = 0
        while empt and len(pos) < n:
            pos.append((b[0], b[1], turn))
            c = empt[rng.randrange(len(empt))]
            empt[empt.index(c)] = empt[-1]
            empt.pop()
            b[turn] |= 1 << c
            if has_five(b[turn]):
                break
            turn ^= 1
    return pos



def main():
    pos = sample(N)
    print(f"positions: {len(pos)}")

    # ceiling: a position can never be a VCF win if the mover has no four at all
    has_force = 0
    for bb0, bb1, turn in pos:
        att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
        if forcing_moves(att, dfn):
            has_force += 1
    print(f"positions where the mover has >=1 forcing move (VCF ceiling): "
          f"{has_force/len(pos):.1%}")
    print()

    print(f"{'depth':>6} {'proven +1':>10} {'of ceiling':>11} {'ms/pos':>9}")
    print(" " + "-" * 40)
    prev = None
    for d in DEPTHS:
        t0 = time.perf_counter()
        hits = 0
        for bb0, bb1, turn in pos:
            att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
            if vcf(att, dfn, d):
                hits += 1
        dt = time.perf_counter() - t0
        rate = hits / len(pos)
        gain = "" if prev is None else f"  (+{rate-prev:.1%})"
        print(f"{d:>6} {rate:>9.1%} {rate/(has_force/len(pos)):>10.1%} "
              f"{1000*dt/len(pos):>8.2f}{gain}")
        prev = rate

    # --- the negative side: depth should close proofs much faster here --------
    print()
    print("negative labels (all moves must lose) at two depths:")
    sub = pos[:250]
    for d in (3, 7):
        t0 = time.perf_counter()
        neg = 0
        for bb0, bb1, turn in sub:
            att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
            if vcf(att, dfn, d):
                continue
            occ = att | dfn
            cand = _near_stones(occ)
            if not cand:
                continue
            all_lose = True
            for m in cand:
                a2 = att | (1 << m)
                if has_five(a2):
                    all_lose = False
                    break
                if not vcf(dfn, a2, d):
                    all_lose = False
                    break
            if all_lose:
                neg += 1
        dt = time.perf_counter() - t0
        print(f"  depth {d:2d}: {neg/len(sub):6.2%}   {1000*dt/len(sub):7.1f} ms/position")


if __name__ == "__main__":
    main()
