"""What does top-k sampling cost in a deterministic game?

LLM-style top-k works because a slightly-off token still yields coherent text.
In gomoku a slightly-off move can simply lose. This measures the price directly:
same network, same openings, only the move-selection rule changes.
"""

import paths
import random
import sys

import torch

from engine import CELLS, has_five
from play import GreedyAgent, SearchAgent, load_net

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PLIES = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TAG = sys.argv[3] if len(sys.argv) > 3 else "v2"
KS = [1, 2, 3, 5, 10]


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
    net, ck = load_net(paths.run_ckpt(TAG), dev)
    openings = make_openings(N, 4)
    greedy = GreedyAgent()
    print(f"network {TAG}, depth {PLIES}, {N} openings shared by every rule")
    print(f"{'selection rule':<22}{'vs greedy':>11}{'delta':>10}")
    print("-" * 44)
    base = None
    for k in KS:
        rules = {"topk": k} if k > 1 else {}
        a = SearchAgent(net, dev, PLIES, **rules)
        res = wins_against(a, greedy, openings)
        wr = sum(res) / N
        if base is None:
            base, extra = res, ""
        else:
            m, s = paired(res, base)
            extra = f"{m*100:+.1f} pt"
        print(f"top-{k} uniform{'':<9}{wr:>10.1%}{extra:>10}")
    print()
    print("top-1 = argmax (ties broken by the shuffle).")


if __name__ == "__main__":
    main()
