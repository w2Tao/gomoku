"""Is 2-ply usable now that the value is better?

The earlier failure had a measured cause: the error of a single value estimate
(~0.24) was larger than the signal across my candidate moves (~0.088), so any
operation that collapsed the opponent's ~45 replies into one number was decided
by noise. This re-measures both sides of that ratio with baseline1, and then
just plays depth 1 against depth 2.

    python code/depth_check.py 100
"""
from __future__ import annotations

import random
import sys

import numpy as np
import torch

import baselines
import paths
from engine import CELLS, has_five
from play import GreedyPlusAgent, SearchAgent

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
AGG = sys.argv[2] if len(sys.argv) > 2 else "min"
OPENING = 4
BID = "baseline2"


def sample_positions(d, n=40, seed=5):
    rng = random.Random(seed)
    idx = list(range(len(d["counts"])))
    rng.shuffle(idx)
    return idx[:n]


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
    net, man = baselines.load_baseline(BID, dev)
    from dataset import bitboards, load
    d = load(paths.data("gomocup15.npz"))
    own, opp = bitboards(d)
    idx = sample_positions(d)

    agent2 = SearchAgent(net, dev, plies=2, agg=AGG)
    cap = {}
    orig = agent2._aggregate

    def spy(v, k, i_win, t_win, legal, their, occ, v_child=None):
        cap["v"] = v.copy()
        cap["k"] = k
        return orig(v, k, i_win, t_win, legal, their, occ, v_child)

    agent2._aggregate = spy
    rng = random.Random(11)

    print("=" * 66)
    print("A. is the value signal still smaller than its own noise?")
    print("=" * 66)
    print(f"{'k':>5}{'sd over my moves':>19}{'sd over replies':>17}"
          f"{'distinct Q':>12}{'tied at top':>13}")
    print("-" * 66)
    sd_mine, sd_reps, dis, ties = [], [], [], []
    for i in idx:
        agent2.move(own[i], opp[i], int(d["last"][i]), rng)
        if "v" not in cap:
            continue
        v, k = cap["v"], cap["k"]
        off = v[~np.eye(k, dtype=bool)].reshape(k, k - 1)
        sd_mine.append(off.var(axis=0).mean() ** 0.5)
        sd_reps.append(off.var(axis=1).mean() ** 0.5)
        q = agent2.last_q
        dis.append(len(np.unique(np.round(q, 6))))
        ties.append(int((np.abs(q - q.max()) < 1e-9).sum()))
        cap.clear()
    print(f"{'-':>5}{np.mean(sd_mine):>19.4f}{np.mean(sd_reps):>17.4f}"
          f"{np.mean(dis):>12.1f}{np.mean(ties):>13.1f}")
    print()
    print("reference, measured on the old model: my moves 0.088, replies 0.135,")
    print("distinct Q 12.1, tied at top 6.9, single-value error ~0.24")

    print()
    print("=" * 66)
    print("B. play depth 1 against depth 2")
    print("=" * 66)
    openings = make_openings(N, OPENING)
    ref = GreedyPlusAgent()
    a1 = SearchAgent(net, dev, plies=1)
    a2 = SearchAgent(net, dev, plies=2, agg=AGG)
    s1 = scores(a1, ref, openings)
    s2 = scores(a2, ref, openings)
    print(f"{N} openings x 2 colours = {2*N} games each, vs greedy+")
    print(f"  depth 1 : {sum(s1)/N:6.1%}")
    print(f"  depth 2 : {sum(s2)/N:6.1%}")
    m, s = paired(s2, s1)
    print(f"  depth2 over depth1: {m:+.1%} ({s:+.1f} sigma, paired)")


if __name__ == "__main__":
    main()
