"""How much does each upgrade actually buy?

Same paired openings everywhere, so the numbers are comparable:

    greedy+                    three plies, one double threat, random otherwise
    heuristic (score only)     + a static window score, no VCF
    heuristic (vcf + score)    + a depth-7 VCF search
"""
import random
import sys

import torch

import paths
from engine import CELLS, has_five
from play import (GreedyPlusAgent, HeuristicAgent, PolicyAgent, SearchAgent,
                  load_net, load_policy_net)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OPENING = 4
TAG = sys.argv[2] if len(sys.argv) > 2 else "pi1"


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


def score(a, b, openings, seed=77):
    rng = random.Random(seed)
    w = 0
    for seq in openings:
        w += play_from(a, b, seq, rng) == 0
        w += play_from(b, a, seq, rng) == 1
    return w / (2 * len(openings))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    openings = make_openings(N, OPENING)

    full = HeuristicAgent(use_vcf=True, use_score=True)
    noVcf = HeuristicAgent(use_vcf=False, use_score=True)
    noScore = HeuristicAgent(use_vcf=True, use_score=False)
    gplus = GreedyPlusAgent()

    pnet, _ = load_policy_net(f"{paths.run_dir(TAG)}/best.pt", dev)
    policy = PolicyAgent(pnet, dev)
    vnet, _ = load_net(paths.baseline_dir("baseline0") + "/model.pt", dev)
    teacher = SearchAgent(vnet, dev, plies=1)

    tests = [
        ("heuristic (vcf+score)", full, "greedy+", gplus),
        ("heuristic (score only)", noVcf, "greedy+", gplus),
        ("heuristic (vcf only)", noScore, "greedy+", gplus),
        ("heuristic (vcf+score)", full, "policy net", policy),
        ("heuristic (vcf+score)", full, "1-ply search", teacher),
        ("greedy+", gplus, "1-ply search", teacher),
    ]
    print(f"{N} openings x 2 colours = {2*N} games per row, "
          f"{OPENING} random opening plies")
    print()
    print(f"{'A':<24}{'B':<14}{'A win rate':>12}")
    print("-" * 50)
    for name_a, a, name_b, b in tests:
        print(f"{name_a:<24}{name_b:<14}{score(a, b, openings):>11.1%}",
              flush=True)


if __name__ == "__main__":
    main()
