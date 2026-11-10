"""Compare the three trained models by playing them against each other.

Training metrics are not comparable across teachers: the heuristic is
deterministic, so its moves have one right answer and accuracy looks high, while
engine moves are diverse and any of several is fine. Only games settle it.

Each model is played through its policy head alone -- a single forward pass,
argmax over legal cells -- so the comparison measures the learned policy, not a
search wrapper around it.

    python code/eval_three.py 100
"""
from __future__ import annotations

import random
import sys

from engine import CELLS, has_five
from play import PolicyAgent, frozen_heuristic, load_policy_net

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OPENING = 4

RUNS = {
    "t1 teacher": "t1_teacher",
    "t2 finetune": "t2_finetune",
    "t3 gomocup": "t3_gomocup",
}


def make_openings(k, opening, seed=99):
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


def score(a, b, openings, seed=77):
    rng = random.Random(seed)
    out = []
    for seq in openings:
        w = 0
        w += play_from(a, b, seq, rng) == 0
        w += play_from(b, a, seq, rng) == 1
        out.append(w / 2.0)
    return out


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    agents = {}
    for name, tag in RUNS.items():
        net, _ = load_policy_net(f"ckpt/runs/{tag}/best.pt", dev)
        agents[name] = PolicyAgent(net, dev)
    agents["frozen heuristic"] = frozen_heuristic()

    openings = make_openings(N, OPENING)
    names = list(RUNS)

    print(f"{N} openings x 2 colours = {2*N} games per matchup, "
          f"{OPENING} random opening plies")
    print()
    print(f"{'agent':<16}{'vs heuristic':>14}{'vs t1':>9}{'vs t2':>9}{'vs t3':>9}")
    print("-" * 58)

    table = {}
    for a in names:
        row = {}
        row["frozen heuristic"] = sum(
            score(agents[a], agents["frozen heuristic"], openings)) / N
        for b in names:
            if b == a:
                row[b] = None
            else:
                row[b] = sum(score(agents[a], agents[b], openings)) / N
        table[a] = row

    for a in names:
        r = table[a]
        cells = []
        for b in ["frozen heuristic"] + names:
            v = r[b]
            cells.append(f"{v:>8.1%}" if v is not None else f"{'—':>8}")
        print(f"{a:<16}" + "".join(cells))
    print()
    print("(a row reads: that agent's score against each column opponent)")


if __name__ == "__main__":
    main()
