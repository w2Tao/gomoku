"""Paired, statistically valid strength comparison.

Two fixes over the naive harness:

  * the first `opening` plies of every game are random, so the games are actually
    distinct. A deterministic agent otherwise replays one game N times and
    "40 games" is a single sample.
  * every network faces the *same* set of openings, so the comparison is paired
    and the first-player advantage cancels out.

    python paired_eval.py 200 1 4 v1,v2
"""

import paths
import math
import random
import sys

import torch

from engine import CELLS, has_five
from model import ValueNet
from play import GreedyAgent, RandomAgent, SearchAgent, load_net

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PLIES = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OPENING = int(sys.argv[3]) if len(sys.argv) > 3 else 4
TAGS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["v1", "v2"]


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


def wins_against(net, opp, openings, plies, dev):
    a = SearchAgent(net, dev, plies)
    rng = random.Random(7)
    out = []
    for seq in openings:
        w = 0
        w += play_from(a, opp, seq, rng) == 0
        w += play_from(opp, a, seq, rng) == 1
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
    nets, cfg = {}, None
    for tag in TAGS:
        net, ck = load_net(paths.run_ckpt(tag), dev)
        nets[tag] = net
        cfg = ck["config"]
    torch.manual_seed(123)
    nets["random-init"] = ValueNet(**cfg).to(dev).eval()

    openings = make_openings(N, OPENING)
    print(f"depth {PLIES} ply | {N} openings x 2 colours = {2*N} games "
          f"| {OPENING} random opening plies")
    print()
    for opp_name, opp in (("greedy", GreedyAgent()),):
        print(f"vs {opp_name}")
        res = {}
        for tag in TAGS + ["random-init"]:
            res[tag] = wins_against(nets[tag], opp, openings, PLIES, dev)
            print(f"   {tag:<12} {sum(res[tag])/N:6.1%}")
        for tag in TAGS:
            m, s = paired(res[tag], res["random-init"])
            print(f"   {tag} over random-init: {m:+.1%} ({s:+.1f} sigma, paired)")
        if len(TAGS) >= 2:
            m, s = paired(res[TAGS[-1]], res[TAGS[0]])
            print(f"   {TAGS[-1]} over {TAGS[0]}: {m:+.1%} ({s:+.1f} sigma, paired)")
        print()


if __name__ == "__main__":
    main()
