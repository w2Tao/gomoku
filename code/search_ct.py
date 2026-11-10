"""Does a policy-pruned 2-ply negamax beat the policy head on its own?

A 1-ply agent that takes argmax(-V(child)) over every legal cell lost 23.5 points:
one value error decides every candidate. This tests the obvious repair -- give the
opponent a reply, order candidates by the policy prior, and let the value head
score only a handful of grandchildren.

    python code/search_ct.py baseline 120 --width 12 --replies 8
    python code/search_ct.py baseline 120 --width 24 --replies 12
"""
from __future__ import annotations

import argparse
import random

import torch

import baselines
from engine import CELLS, has_five
from play import (HybridAgent, PolicyAgent, PolicySearchAgent,
                  frozen_heuristic)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", nargs="?", default="baseline")
    ap.add_argument("openings", nargs="?", type=int, default=120)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--replies", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, man = baselines.load_baseline(args.baseline, dev)
    policy = PolicyAgent(net, dev)
    search = PolicySearchAgent(net, dev, width=args.width,
                               replies=args.replies)
    hybrid = HybridAgent(net, dev)
    ref = frozen_heuristic()

    openings = make_openings(args.openings)
    n = len(openings)
    print(f"{args.baseline} ({man['parameters']:,} params)")
    print(f"{n} openings x 2 colours = {2*n} games per row")
    print(f"search: 2-ply, width {args.width}, replies {args.replies}\n")

    s_pol = scores(policy, ref, openings)
    s_srch = scores(search, ref, openings)
    s_hyb = scores(hybrid, ref, openings)
    print(f"{'agent':<26}{'vs heuristic':>14}")
    print("-" * 42)
    print(f"{'policy head only':<26}{sum(s_pol)/n:>13.1%}")
    print(f"{'policy + 2-ply search':<26}{sum(s_srch)/n:>13.1%}")
    print(f"{'hybrid (tactics + policy)':<26}{sum(s_hyb)/n:>13.1%}")

    d = [a - b for a, b in zip(s_srch, s_pol)]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)
    se = max(var ** 0.5 / len(d) ** 0.5, 1e-9)
    print(f"\nsearch over policy: {m:+.1%} ({m/se:+.1f} sigma, paired)")

    d = [a - b for a, b in zip(s_hyb, s_pol)]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)
    se = max(var ** 0.5 / len(d) ** 0.5, 1e-9)
    print(f"hybrid over policy: {m:+.1%} ({m/se:+.1f} sigma, paired)")

    h = scores(search, policy, openings, seed=777)
    print(f"head to head, search vs policy: {sum(h)/n:.1%}")
    h2 = scores(hybrid, policy, openings, seed=555)
    print(f"head to head, hybrid vs policy: {sum(h2)/n:.1%}")


if __name__ == "__main__":
    main()
