"""Does an opening specialist plus an in-game specialist beat one model?

Opening play and in-game play may not be the same skill, and the corpus is not
neutral about the split: after deduplication only 5% of positions are in the first
ten plies, but the distillation data is rich there (33,763 rows at ply <= 10, versus
17,292 in the corpus) because it is collected from randomised openings. So the
opening half has its own data source.

The question the games answer is whether handing over at ply N beats never handing
over. `switch=0` means the in-game model plays everything, which is the control.

    python code/handoff_eval.py d3 open1 120
"""
from __future__ import annotations

import argparse
import random
import sys

import torch

sys.path.insert(0, "code")

import baselines                                       # noqa: E402
from engine import CELLS, has_five                     # noqa: E402
from play import PolicyAgent, frozen_heuristic         # noqa: E402


class Handoff:
    """Opening net until `switch` stones are down, then the in-game net."""

    def __init__(self, opening_net, ingame_net, dev, switch):
        self.open = PolicyAgent(opening_net, dev) if opening_net else None
        self.game = PolicyAgent(ingame_net, dev)
        self.switch = switch

    def move(self, my, their, last, rng):
        stones = bin(my | their).count("1")
        a = self.open if (self.open is not None and stones < self.switch) else self.game
        return a.move(my, their, last, rng)


def make_openings(k, opening=4, seed=99):
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        bb = [0, 0]
        seq, ok = [], True
        for ply in range(opening):
            legal = [c for c in range(CELLS) if not ((bb[0] | bb[1]) >> c) & 1]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("ingame", nargs="?", default="d3")
    ap.add_argument("opening", nargs="?", default="open1")
    ap.add_argument("openings", nargs="?", type=int, default=120)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gnet, _ = baselines.load_baseline(args.ingame, dev)
    onet, _ = baselines.load_baseline(args.opening, dev)
    ref = frozen_heuristic()

    ops = make_openings(args.openings)
    n = len(ops)
    print(f"{n} openings x 2 colours = {2*n} games per row")
    print(f"in-game: {args.ingame}   opening: {args.opening}\n")
    print(f"{'switch at ply':<16}{'vs heuristic':>14}")
    print("-" * 30)

    base = None
    for sw in (0, 6, 10, 14):
        agent = Handoff(None if sw == 0 else onet, gnet, dev, sw)
        s = sum(score(agent, ref, ops)) / n
        base = base if base is not None else s
        tag = "control" if sw == 0 else f"{sw}"
        print(f"{tag:<16}{s:>13.1%}")
        if sw == 0:
            ctrl_score = s

    # direct head to head: handoff at 10 against the single in-game model
    single = Handoff(None, gnet, dev, 0)
    hand = Handoff(onet, gnet, dev, 10)
    h = score(hand, single, ops, seed=555)
    print(f"\nhead to head, handoff@10 vs {args.ingame}: {sum(h)/n:.1%}")


if __name__ == "__main__":
    main()
