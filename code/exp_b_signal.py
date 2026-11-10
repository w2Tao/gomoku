"""Experiment B: how much information does a raw game-outcome label carry?

For a position s with side-to-move value p = P(mover wins | uniform random play),
the Monte-Carlo target z (the outcome of the one game s was sampled from) is a
Bernoulli(p) draw. Its information content about p is 1 - H2(p) bits, with a
maximum of 1 bit and exactly 0 bits when p = 0.5.

We estimate p by rollout, then report how that information budget is spread
over the length of a 9x9 game.
"""

import math
import random
import sys
import statistics

from engine import CELLS, has_five, playout

GAMES = 50
ROLLOUTS = 2000
PLIES = list(range(0, 55, 3))

argv = __import__("sys").argv
if len(argv) > 1:
    GAMES = int(argv[1])
if len(argv) > 2:
    ROLLOUTS = int(argv[2])


def h2(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def one_game_states(rng):
    """All (bb0, bb1, turn) states, plus the winner (None on draw)."""
    b = [0, 0]
    empt = list(range(CELLS))
    turn = 0
    hist = []
    while True:
        if not empt:
            return hist, None
        hist.append((b[0], b[1], turn))
        k = rng.randrange(len(empt))
        cell = empt[k]
        empt[k] = empt[-1]
        empt.pop()
        b[turn] |= 1 << cell
        if has_five(b[turn]):
            return hist, turn
        turn ^= 1


def mc_value(bb0, bb1, turn, n, rng):
    w = 0
    for _ in range(n):
        r = playout(bb0, bb1, turn, rng)
        if r == turn:
            w += 1
    return w / n



def main():
    rng = random.Random(2024)

    # collect states, one bucket per ply
    buckets = {t: [] for t in PLIES}
    for _ in range(GAMES):
        hist, winner = one_game_states(rng)
        for t in PLIES:
            if t < len(hist):
                buckets[t].append((hist[t], winner))

    print(f"games={GAMES}  rollouts/position={ROLLOUTS}")
    print()
    print(" ply   n   mean p   std p   |  mean info(bits)  frac in [0.4,0.6]")
    print(" ────────────────────────────────────────────────────────────────")
    rows = []
    for t in PLIES:
        vals, infos = [], []
        for (bb0, bb1, turn), _winner in buckets[t]:
            p = mc_value(bb0, bb1, turn, ROLLOUTS, rng)
            vals.append(p)
            infos.append(1 - h2(p))
        if not vals:
            continue
        rows.append((t, vals, infos))
        near = sum(1 for v in vals if 0.4 <= v <= 0.6) / len(vals)
        print(f" {t:3d} {len(vals):4d}   {statistics.mean(vals):.3f}   "
              f"{statistics.stdev(vals):.3f}   |     {statistics.mean(infos):.3f}          {near:.2f}")

    total_positions = sum(len(v) for _, v, _ in rows)
    all_info = [i for _, _, inf in rows for i in inf]
    print()
    print(f"positions sampled          : {total_positions}")
    print(f"mean information per label : {statistics.mean(all_info):.4f} bits  (max 1.0)")
    print(f"labels above 0.1 bits      : {sum(1 for i in all_info if i > 0.1)/len(all_info):.1%}")

    # The last-plies regime: value collapses to +-1 only right at the end.
    print()
    print("information budget by phase (share of the 1-bit ceiling):")
    for t, v, inf in rows:
        bar = "#" * int(round(statistics.mean(inf) * 50))
        print(f"  ply {t:2d}: {statistics.mean(inf):.3f} bits  {bar}")


if __name__ == "__main__":
    main()
