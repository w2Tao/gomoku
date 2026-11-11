"""Does a policy-prior search beat the policy head of a larger base model?

This is the retry of the search experiment in report.md section 7.7. The earlier
2-ply negamax lost because the value head was too coarse to rank moves; the v2
base now has a much larger trunk and a value head trained on game outcomes
before the distillation pass. The question is empirical, so it gets its own
paired-opening harness instead of a training metric.

    python code/search_eval.py v2base 60 --width 12 --replies 8

Both agents are deterministic; the first four plies are random and every
opening is played twice with colours swapped, so the comparison is paired.

`--value-model` keeps the policy net for move ordering but scores leaves with a
different checkpoint (the outcome-trained base, whose value head survives
distillation). It switches to `SplitPolicySearchAgent`.
"""
from __future__ import annotations

import argparse
import random

import torch

import baselines
from engine import CELLS, has_five
from play import (PolicyAgent, PolicySearchAgent, SplitPolicySearchAgent,
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
    """Per-opening score of `a` against `b`, colours pooled."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="v2base")
    ap.add_argument("openings", nargs="?", type=int, default=60)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--replies", type=int, default=8)
    ap.add_argument("--plies", type=int, default=2, choices=(1, 2),
                    help="1 = policy-pruned value ranking; 2 = two-ply search")
    ap.add_argument("--value-model", default=None,
                    help="separate value net, e.g. v2s1")
    ap.add_argument("--scalar-value", action="store_true",
                    help="value model is a ValuePolicyEvalNet; use its scalar head")
    args = ap.parse_args()
    if args.plies == 1 and not args.value_model:
        raise SystemExit("--plies 1 needs --value-model (the split agent)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pnet, man = baselines.load_baseline(args.model, dev)
    if man.get("model_type") != "value_policy":
        raise SystemExit(f"{args.model} has no policy head")
    policy = PolicyAgent(pnet, dev)
    if args.value_model:
        vnet, vman = baselines.load_baseline(args.value_model, dev)
        search = SplitPolicySearchAgent(pnet, vnet, dev, width=args.width,
                                        replies=args.replies,
                                        plies=args.plies,
                                        scalar_value=args.scalar_value)
        search_name = f"policy {args.model} + value {args.value_model}"
    else:
        vman = man
        search = PolicySearchAgent(pnet, dev, width=args.width,
                                   replies=args.replies)
        search_name = f"policy + value {args.model}"
    ref = frozen_heuristic()
    ops = make_openings(args.openings)
    n = len(ops)

    print(f"{args.model}: {man['parameters']:,} params, {dev}")
    print(f"value: {vman['parameters']:,} params  ({search_name})")
    print(f"{n} openings x 2 colours = {2*n} games per row")
    print(f"search: {args.plies}-ply, width {args.width}, "
          f"replies {args.replies}\n")

    s_pol = scores(policy, ref, ops)
    s_srch = scores(search, ref, ops)
    print(f"{'agent':<26}{'vs heuristic':>14}")
    print("-" * 42)
    print(f"{'policy head only':<26}{sum(s_pol)/n:>13.1%}")
    print(f"{f'policy + {args.plies}-ply search':<26}{sum(s_srch)/n:>13.1%}")

    m, s = paired(s_srch, s_pol)
    print(f"\nsearch over policy, same openings: {m:+.1%} ({s:+.1f} sigma)")

    h = scores(search, policy, ops, seed=555)
    print(f"head to head, search vs policy: {sum(h)/n:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
