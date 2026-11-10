"""Playing-strength duel between frozen baselines.

Each baseline is loaded from its manifest, so the inference recipe travels with
the artifact. The comparison is paired on openings (identical starting positions
for every agent) and reported both head-to-head and against a common reference
opponent -- head-to-head alone is a weak signal between two similar agents,
because the games are few and the error bars wide.

    python code/baseline_duel.py 200
"""
from __future__ import annotations

import math
import random
import sys

import torch

import baselines
import paths
from engine import CELLS, has_five
from play import GreedyPlusAgent, HeuristicAgent, SearchAgent

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
OPENING = 4
IDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["baseline"]


def make_openings(k, opening, seed=99):
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        bb = [0, 0]
        seq, ok = [], True
        for ply in range(opening):
            turn = ply % 2
            occ = bb[0] | bb[1]
            legal = [c for c in range(CELLS) if not (occ >> c) & 1]
            m = rng.choice(legal)
            bb[turn] |= 1 << m
            seq.append(m)
            if has_five(bb[turn]):
                ok = False
                break
        if ok:
            out.append(seq)
    return out


def play_from(black, white, seq, rng):
    bb = [0, 0]
    last = -1
    for ply, m in enumerate(seq):
        bb[ply % 2] |= 1 << m
        last = m
    turn = len(seq) % 2
    for _ in range(CELLS):
        agent = black if turn == 0 else white
        m = agent.move(bb[turn], bb[turn ^ 1], last, rng)
        if m is None:
            return None
        bb[turn] |= 1 << m
        last = m
        if has_five(bb[turn]):
            return turn
        turn ^= 1
    return None


def scores(a, b, openings, seed=77):
    """Per-opening score of `a` against `b`, colours pooled (0 / 0.5 / 1)."""
    rng = random.Random(seed)
    out = []
    for seq in openings:
        w = 0
        w += play_from(a, b, seq, rng) == 0
        w += play_from(b, a, seq, rng) == 1
        out.append(w / 2.0)
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)
    se = max(var ** 0.5 / len(d) ** 0.5, 1e-9)
    return m, m / se


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    agents = {}
    for bid in IDS:
        net, man = baselines.load_baseline(bid, dev)
        depth = man.get("inference", {}).get("search_depth_plies", 1) or 1
        agents[bid] = SearchAgent(net, dev, plies=depth)
        print(f"{bid:<12} depth {depth}  ->  {man['name']}")
    ref = GreedyPlusAgent()
    openings = make_openings(N, OPENING)
    print()
    print(f"{N} openings x 2 colours = {2*N} games per matchup, "
          f"{OPENING} random opening plies")
    print()

    res = {}
    print(f"{'agent':<14}{'vs greedy+':>12}{'sigma vs base':>16}")
    print("-" * 44)
    for bid in IDS:
        res[bid] = scores(agents[bid], ref, openings)
        print(f"{bid:<14}{sum(res[bid])/N:>11.1%}")
    if len(IDS) >= 2:
        base = res[IDS[0]]
        for bid in IDS[1:]:
            m, s = paired(res[bid], base)
            print(f"  {bid} over {IDS[0]}: {m:+.1%} ({s:+.1f} sigma, paired)")

    if len(IDS) >= 2:
        print()
        h = scores(agents[IDS[1]], agents[IDS[0]], openings, seed=777)
        print(f"head to head {IDS[1]} vs {IDS[0]}: {IDS[1]} scores "
              f"{sum(h)/N:.1%}")
        hb = scores(agents[IDS[0]], agents[IDS[1]], openings, seed=777)
        m, s = paired(h, hb)
        print(f"  colour-swapped consistency check: {m:+.1%} ({s:+.1f} sigma)")


if __name__ == "__main__":
    main()
