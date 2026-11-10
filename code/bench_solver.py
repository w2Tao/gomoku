"""Throughput of the forced-win detector -- i.e. what it actually costs.

There are no parameters here and nothing is learned: the cost is pure search
time. This measures how many positions per second a 3-ply detector can label.
"""

import random
import time

from engine import CELLS, has_five
from solvers import win_in_1, win_in_3, five_cells, _near_stones

N_POS = 400


def sample_positions(rng, n_games):
    pos = []
    for _ in range(n_games):
        b = [0, 0]
        empt = list(range(CELLS))
        turn = 0
        while empt and len(pos) < 10000:
            if rng.random() < 0.15:
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
    rng = random.Random(11)
    pos = sample_positions(rng, 200)[:N_POS]
    print(f"sampled {len(pos)} positions")

    t0 = time.perf_counter()
    n1 = 0
    for bb0, bb1, turn in pos:
        att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
        n1 += win_in_1(att, att | dfn)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    n3 = 0
    for bb0, bb1, turn in pos:
        att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
        n3 += win_in_3(att, dfn)
    t3 = time.perf_counter()

    n = len(pos)
    print(f"win_in_1 : {t1-t0:.3f}s  -> {n/(t1-t0):8,.0f} positions/s   ({n1} hits)")
    print(f"win_in_3 : {t3-t2:.3f}s  -> {n/(t3-t2):8,.0f} positions/s   ({n3} hits)")
    print(f"cost of going 1 -> 3 plies: {(t3-t2)/(t1-t0):.1f}x")

    # average size of the forcing-move frontier, which drives deeper search cost
    sizes = []
    for bb0, bb1, turn in pos[:100]:
        att, dfn = (bb0, bb1) if turn == 0 else (bb1, bb0)
        sizes.append(len(_near_stones(att | dfn)))
    print(f"\ncandidate moves per position: mean {sum(sizes)/len(sizes):.1f}, "
          f"max {max(sizes)}")
    print("deeper VCF keeps only four-making moves, so its branching is far smaller")
    print("than the candidate list -- that is why depth 5-9 stays affordable.")


if __name__ == "__main__":
    main()
