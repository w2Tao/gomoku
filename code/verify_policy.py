"""Independent verification of the policy result.

A distilled policy beating its teacher by 13.5 points is not a normal outcome, so
before believing it: re-run with different opening/game seeds, check that every
move is legal, count how often the teacher's argmax is decided by a tie, and
measure agreement between the policy and the teacher.
"""
import random
import sys

import numpy as np
import torch

import paths
from engine import CELLS, has_five
from model import ValuePolicyNet
from play import GreedyAgent, PolicyAgent, SearchAgent, load_net, load_policy_net

TAG = sys.argv[1] if len(sys.argv) > 1 else "pi1"


def make_openings(k, opening, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        bb = [0, 0]
        seq = []
        ok = True
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
    seen = set(seq)
    for _ in range(CELLS):
        agent = black if turn == 0 else white
        m = agent.move(bb[turn], bb[turn ^ 1], last, rng)
        if m is None:
            return None
        assert 0 <= m < CELLS and m not in seen and not (bb[turn] >> m) & 1, \
            f"illegal move {m}"
        seen.add(m)
        bb[turn] |= 1 << m
        last = m
        if has_five(bb[turn]):
            return turn
        turn ^= 1
    return None


def wins_against(agent, opp, openings, seed):
    rng = random.Random(seed)
    out = []
    for seq in openings:
        w = 0
        w += play_from(agent, opp, seq, rng) == 0
        w += play_from(opp, agent, seq, rng) == 1
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
    pnet, ck = load_policy_net(f"{paths.run_dir(TAG)}/best.pt", dev)
    vnet, _ = load_net(paths.baseline_dir("baseline0") + "/model.pt", dev)
    policy = PolicyAgent(pnet, dev)
    teacher = SearchAgent(vnet, dev, plies=1)
    greedy = GreedyAgent()

    N, OPN, SEED = 300, 5, 12345
    openings = make_openings(N, OPN, SEED)
    print(f"independent run: {N} openings (seed {SEED}), {OPN} random plies, "
          f"game seed 4242")

    res_p = wins_against(policy, greedy, openings, 4242)
    res_t = wins_against(teacher, greedy, openings, 4242)
    print(f"  policy  vs greedy : {sum(res_p)/N:.1%}")
    print(f"  teacher vs greedy : {sum(res_t)/N:.1%}")
    m, s = paired(res_p, res_t)
    print(f"  policy over teacher: {m:+.1%} ({s:+.1f} sigma)")

    h2h = wins_against(policy, teacher, openings, 777)
    print(f"  head to head      : policy {sum(h2h)/N:.1%} vs teacher")

    # --- how often is the teacher's choice a tie? ---------------------
    rng = random.Random(3)
    ties, agree, n = [], 0, 0
    for seq in openings[:120]:
        bb = [0, 0]
        last = -1
        for ply, mm in enumerate(seq):
            bb[ply % 2] |= 1 << mm
            last = mm
        turn = len(seq) % 2
        my, their = bb[turn], bb[turn ^ 1]
        q = teacher.move(my, their, last, rng)
        teacher.last_q is not None and ties.append(
            int((np.abs(teacher.last_q - teacher.last_q.max()) < 1e-9).sum()))
        pm = policy.move(my, their, last, rng)
        agree += int(pm == q)
        n += 1
    print()
    print(f"  teacher's Q ties at the max : mean {np.mean(ties):.1f} of "
          f"{CELLS - len(seq):.0f} moves")
    print(f"  policy agrees with teacher  : {agree/n:.1%}")


if __name__ == "__main__":
    main()
