"""Experiment G: label distribution as a function of VCF depth.

Same sampling as exp_d, but the positive and negative sides both use the real
depth-limited VCF. This is the number that says whether depth is worth buying.
"""

import collections
import random
import sys
import time

from engine import CELLS, has_five
from solvers import _near_stones, win_in_1, win_in_3
from vcf import vcf

DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 7
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def sample(rng, n_games):
    pos = []
    for _ in range(n_games):
        b = [0, 0]
        empt = list(range(CELLS))
        turn = 0
        while empt:
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
    rng = random.Random(4242)
    pos = sample(rng, GAMES)
    rec = collections.defaultdict(collections.Counter)
    t0 = time.time()
    for bb0, bb1, turn in pos:
        att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
        ply = bin(bb0).count("1") + bin(bb1).count("1")
        if vcf(att, dfn, DEPTH):
            lab = 1
        else:
            occ = att | dfn
            cand = _near_stones(occ)
            lab = None
            if cand:
                lab = -1
                for m in cand:
                    a2 = att | (1 << m)
                    if has_five(a2) or not vcf(dfn, a2, DEPTH):
                        lab = None
                        break
        rec[(ply // 10) * 10][lab] += 1
    dt = time.time() - t0

    tot = collections.Counter()
    for b in rec:
        tot.update(rec[b])
    n = len(pos)

    print(f"VCF depth={DEPTH}  games={GAMES}  positions={n}  ({n/dt:,.0f} pos/s)")
    print()
    print(" ply     n     +1       -1     unknown")
    print(" " + "-" * 38)
    for b in sorted(rec):
        c = rec[b]
        m = sum(c.values())
        print(f" {b:2d}-{b+9:2d} {m:5d}  {c[1]/m:6.2%}  {c[-1]/m:6.2%}  {c[None]/m:7.2%}")
    print(" " + "-" * 38)
    print(f" ALL   {n:5d}  {tot[1]/n:6.2%}  {tot[-1]/n:6.2%}  {tot[None]/n:7.2%}")
    print()
    print(f"usable for training : {(tot[1]+tot[-1])/n:.2%}")
    print(f"+1 : -1 ratio       : {tot[1]/max(tot[-1],1):.1f} : 1")


if __name__ == "__main__":
    main()
