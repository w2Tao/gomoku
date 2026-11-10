"""Does the value head add anything on top of the policy head?

The baseline plays straight off its policy head and throws the value away. Given
0.4-1 ms per value pass and ~45 candidate moves, a 1-ply search costs ~45 forward
passes per move -- cheap enough to test.

    python code/search_ct.py baseline 120
    python code/search_ct.py baseline 120 --plies 2
"""
from __future__ import annotations

import random
import sys

import torch

import baselines
from engine import CELLS, has_five
from play import PolicyAgent, SearchAgent, frozen_heuristic


def make_openings(k, opening=4, seed=99):
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        bb = [0, 0]
        seq, ok = [], True
        for ply in range(opening):
            occ = bb[0] | bb[1]
            legal = [c for c in range(CELLS) if not (occ >> c) & 1]
            m = rng.choice(legal)
            bb[ply % 2] |= 1 << m
            seq.append(m)
            if has_five(bb[ply % 2]):
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
    rng = random.Random(seed)
    out = []
    for seq in openings:
        w = 0
        w += play_from(a, b, seq, rng) == 0
        w += play_from(b, a, seq, rng) == 1
        out.append(w / 2.0)
    return out


def main():
    bid = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    plies = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, man = baselines.load_baseline(bid, dev)
    policy = PolicyAgent(net, dev)
    search = SearchAgent(net, dev, plies=plies, agg="selective" if plies > 1 else "min")
    ref = frozen_heuristic()

    openings = make_openings(n)
    print(f"{bid} ({man['model_type']}, {man['parameters']:,} params)")
    print(f"{n} openings x 2 colours = {2*n} games per row\n")

    s_pol = scores(policy, ref, openings)
    s_srch = scores(search, ref, openings)
    print(f"{'agent':<24}{'vs heuristic':>14}")
    print("-" * 40)
    print(f"{'policy head only':<24}{sum(s_pol)/n:>13.1%}")
    print(f"{'value + %d-ply search' % plies:<24}{sum(s_srch)/n:>13.1%}")

    d = [a - b for a, b in zip(s_srch, s_pol)]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)
    se = max(var ** 0.5 / len(d) ** 0.5, 1e-9)
    print(f"\nsearch over policy: {m:+.1%} ({m/se:+.1f} sigma, paired)")


if __name__ == "__main__":
    main()
