"""Diagnose the policy head: is it genuinely stronger, or just eating greedy's lunch?

Three probes, because a 96% win rate against one weak opponent proves very
little on its own.

  A  colour split      : a policy that only wins as black is riding the first
                         move, not playing well.
  B  stronger opponent : a heuristic that also makes its own double threats and
                         blocks the opponent's. If the policy's win rate collapses
                         against it, the advantage was specific to the old bot.
  C  solver referee    : opponent-free. On held-out positions where the solver
                         proves a win, does the chosen move still preserve it?
                         This is the only probe that uses ground truth rather
                         than relative strength.
"""
import random
import sys

import numpy as np
import torch

import paths
from engine import CELLS, has_five
from labeler import proven_loss
from model import ValuePolicyNet
from play import (GreedyAgent, GreedyPlusAgent, HeuristicAgent, PolicyAgent,
                  SearchAgent, load_net, load_policy_net)
from solvers import WINDOWS_AT, _five_cells_window_scan

TAG = sys.argv[1] if len(sys.argv) > 1 else "pi1"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
OPENING = 4


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


def match_by_colour(a, b, openings, seed=77):
    """(a wins as black, a wins as white, totals)."""
    rng = random.Random(seed)
    aw = bw = 0
    for seq in openings:
        aw += play_from(a, b, seq, rng) == 0
        bw += play_from(b, a, seq, rng) == 1
    return aw, bw, len(openings)


def probe_c(agents, n_pos=120):
    """Solver as referee. agents: [(name, agent)]. Returns ({name: keeps}, same, n).

    The referee MUST use the same depth the labels were built with. `labeler.
    proven_loss` bottoms out in `win_in_3`, a three-ply check, while the dataset
    was labelled with a depth-7 VCF -- using it here would understate every
    agent, including the heuristic, and make the comparison meaningless.
    """
    from build_dataset import label
    from dataset import bitboards, load
    d = load(paths.data("batch1.npz"))
    own, opp = bitboards(d)
    cls = d["cls"]
    is_val = d["is_val"]
    cand = [i for i in range(len(own)) if is_val[i] and cls[i] == 2]
    rng = random.Random(5)
    rng.shuffle(cand)
    cand = cand[:n_pos]

    rng = random.Random(11)
    keep = {name: 0 for name, _ in agents}
    agree = 0
    for i in cand:
        my, their = own[i], opp[i]
        last = int(d["last"][i])
        moves = []
        for name, agent in agents:
            m = agent.move(my, their, last, rng)
            moves.append(m)
            if m is not None and label(their, my | (1 << m), 7) == 0:
                keep[name] += 1
        if len(set(moves)) == 1 and moves[0] is not None:
            agree += 1
    return keep, agree, len(cand)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pnet, ck = load_policy_net(f"{paths.run_dir(TAG)}/best.pt", dev)
    vnet, _ = load_net(paths.baseline_dir("baseline0") + "/model.pt", dev)
    policy = PolicyAgent(pnet, dev)
    teacher = SearchAgent(vnet, dev, plies=1)
    greedy = GreedyAgent()
    gplus = GreedyPlusAgent()
    heur = HeuristicAgent(vcf_depth=7, use_vcf=True, use_score=True)
    openings = make_openings(N, OPENING)

    print("=" * 62)
    print("probe A / B  -  colour split and a stronger opponent")
    print(f"{N} openings x 2 colours per cell, {OPENING} random opening plies")
    print("=" * 62)
    print(f"{'agent':<10}{'vs greedy (b/w)':>24}{'vs greedy+ (b/w)':>24}")
    print("-" * 62)
    for name, a in (("policy", policy), ("teacher", teacher),
                    ("heuristic", heur)):
        gb, gw, _ = match_by_colour(a, greedy, openings)
        pb, pw, _ = match_by_colour(a, gplus, openings)
        print(f"{name:<10}"
              f"{gb/N:>9.0%}/{gw/N:<9.0%}"
              f"{pb/N:>11.0%}/{pw/N:<9.0%}")

    print()
    print("=" * 62)
    print("probe C  -  solver as referee, on held-out positions the solver")
    print("            already proves are won. No opponent involved.")
    print("=" * 62)
    keep, agree, tot = probe_c([("policy", policy), ("teacher", teacher),
                                ("heuristic", heur)])
    print(f"held-out won positions    : {tot}")
    for name in ("policy", "teacher", "heuristic"):
        print(f"{name+' keeps the win':<26}: {keep[name]}/{tot} = "
              f"{keep[name]/tot:.1%}")
    print(f"all three pick the same move: {agree}/{tot} = {agree/tot:.1%}")


if __name__ == "__main__":
    main()
