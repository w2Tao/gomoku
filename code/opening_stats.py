"""Do a few openings dominate because they have known-better win rates?

Reading the corpus by opening line answers two things at once: how concentrated the
opening distribution is, and whether the popular lines are actually scoring better.
If they are not, convergence is a shared convention rather than a discovered edge --
which matters, because a book only pays if the openings in it are genuinely good.

    python code/opening_stats.py [depth]
"""
import glob
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "code")

from build_gomocup_dataset import parse, replay   # noqa: E402

DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def main():
    files = sorted(glob.glob("data/gomocup/Freestyle15_*/*.psq"))
    by_open = defaultdict(lambda: [0, 0])          # key -> [games, black wins]
    total = black_wins = 0
    for path in files:
        moves = parse(path)
        r = replay(moves)
        if r is None:
            continue
        winner, _n = r
        key = tuple((y - 1) * 15 + (x - 1) for x, y in moves[:DEPTH])
        if len(key) < DEPTH:
            continue
        by_open[key][0] += 1
        if winner == 0:
            by_open[key][1] += 1
            black_wins += 1
        total += 1
    if not total:
        sys.exit("no games")

    overall = black_wins / total
    print(f"  games {total:,} | black wins overall {overall:.1%}")
    print(f"  distinct {DEPTH}-move openings: {len(by_open):,}\n")

    ranked = sorted(by_open.items(), key=lambda kv: -kv[1][0])
    covered = 0
    print(f"  {'#moves':>50} {'games':>6} {'share':>7} {'black%':>7}")
    for key, (n, bw) in ranked[:12]:
        covered += n
        seq = " ".join(f"({c % 15},{c // 15})" for c in key)
        print(f"  {seq:>50} {n:>6} {n/total:>6.1%} {bw/n:>6.1%}")
    top = sum(n for _, (n, _b) in ranked[:12])
    print(f"\n  top 12 cover {top/total:.1%} of games")
    print(f"  openings appearing once: {sum(1 for _, (n, b) in ranked if n == 1):,}")

    # is the spread in win rate larger than binomial noise?
    big = [(n, bw) for _k, (n, bw) in ranked if n >= 100]
    if len(big) >= 2:
        rates = sorted(bw / n for n, bw in big)
        import math
        se = math.sqrt(overall * (1 - overall) / 100)
        print(f"\n  openings with >=100 games: {len(big)}")
        print(f"  black win rate range: {rates[0]:.1%} .. {rates[-1]:.1%} "
              f"(spread {rates[-1]-rates[0]:.1%})")
        print(f"  binomial noise at n=100: +/-{1.96*se:.1%}")


if __name__ == "__main__":
    main()
