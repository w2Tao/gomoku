"""Does the distilled policy actually play?

Compares three agents under the same paired openings:

    policy head, one forward pass, argmax      (no search at all)
    value net + 1-ply search                   (the teacher, 85.2%)
    random-init policy head                    (the control)

The control matters for the same reason it did for the value head: a randomly
initialised network would still finish fives and block them via the safety
overrides, so only the gap over that control is attributable to the policy.
"""
import math
import random
import sys

import torch

import paths
from engine import CELLS, has_five
from model import ValuePolicyNet
from play import GreedyAgent, PolicyAgent, SearchAgent, load_net, load_policy_net

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OPENING = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TAG = sys.argv[3] if len(sys.argv) > 3 else "pi1"
TEACHER_BENCH = 0.852


def make_openings(k, opening, seed=99):
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


def wins_against(agent, opp, openings):
    rng = random.Random(7)
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
    ckpt = f"{paths.run_dir(TAG)}/best.pt"
    pnet, ck = load_policy_net(ckpt, dev)
    vnet, _ = load_net(paths.baseline_dir("baseline0") + "/model.pt", dev)

    torch.manual_seed(123)
    ctrl = ValuePolicyNet(**ck["config"]).to(dev).eval()

    agents = {
        "policy (1 fwd)": PolicyAgent(pnet, dev),
        "1-ply search": SearchAgent(vnet, dev, plies=1),
        "policy random-init": PolicyAgent(ctrl, dev),
    }
    openings = make_openings(N, OPENING)
    greedy = GreedyAgent()

    print(f"paired evaluation: {N} openings x 2 colours = {2*N} games")
    print(f"{'agent':<22}{'vs greedy':>11}{'vs teacher 85.2%':>19}")
    print("-" * 52)
    res = {}
    for name, a in agents.items():
        r = wins_against(a, greedy, openings)
        res[name] = r
        wr = sum(r) / N
        print(f"{name:<22}{wr:>10.1%}{wr - TEACHER_BENCH:>+18.1%}")

    print()
    base = res["policy random-init"]
    for name in ("policy (1 fwd)", "1-ply search"):
        m, s = paired(res[name], base)
        print(f"{name} over random-init: {m:+.1%} ({s:+.1f} sigma, paired)")
    m, s = paired(res["policy (1 fwd)"], res["1-ply search"])
    print(f"policy vs its teacher   : {m:+.1%} ({s:+.1f} sigma, paired)")
    print()
    print("a policy that matches the teacher within noise is a usable")
    print("rollout policy: same strength, ~45x cheaper per move.")


if __name__ == "__main__":
    main()
