"""Is a per-move win head a better move ranker than the policy head?

The value head -- one number per position -- was too coarse to order ~150 candidate
moves, and both 1-ply and 2-ply search over it lost to the policy. This tests the
direct alternative: a head that scores every cell with "if I play here, do I win".

Offline it reports how often each head's top choice is the move the engine played,
and how the Q value at the played cell tracks the actual result. On the board it
plays both heads against the frozen heuristic, paired.

    python code/q_eval.py q1 120
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

import paths
from engine import CELLS, H, W, has_five
from model import ValueQNet
from play import frozen_heuristic
from dataset import load, to_planes


class QAgent:
    """Argmax over legal cells of the per-move win head."""

    def __init__(self, net, dev="cpu"):
        self.net = net
        self.dev = dev
        self.last_q = None

    @torch.no_grad()
    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None
        for c in legal:
            if has_five(my | (1 << c)):
                return c
        for c in legal:
            if has_five(their | (1 << c)):
                return c
        x = np.zeros((1, 3, H, W), dtype=np.float32)
        for i, c in enumerate(range(CELLS)):
            if (my >> c) & 1:
                x[0, 0].flat[c] = 1.0
            elif (their >> c) & 1:
                x[0, 1].flat[c] = 1.0
        if last >= 0:
            x[0, 2].flat[last] = 1.0
        _, _, q = self.net(torch.from_numpy(x).to(self.dev))
        q = q[0].cpu().numpy()
        self.last_q = q
        return max(legal, key=lambda c: float(q[c]))


class PolicyAgent2:
    """The same net, but reading its policy head instead."""

    def __init__(self, net, dev="cpu"):
        self.net = net
        self.dev = dev

    @torch.no_grad()
    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None
        for c in legal:
            if has_five(my | (1 << c)):
                return c
        for c in legal:
            if has_five(their | (1 << c)):
                return c
        x = np.zeros((1, 3, H, W), dtype=np.float32)
        for c in range(CELLS):
            if (my >> c) & 1:
                x[0, 0].flat[c] = 1.0
            elif (their >> c) & 1:
                x[0, 1].flat[c] = 1.0
        if last >= 0:
            x[0, 2].flat[last] = 1.0
        _, p, _ = self.net(torch.from_numpy(x).to(self.dev))
        p = p[0].cpu().numpy()
        return max(legal, key=lambda c: float(p[c]))


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
    ap.add_argument("run", nargs="?", default="q1")
    ap.add_argument("openings", nargs="?", type=int, default=120)
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"ckpt/runs/{args.run}/best.pt", map_location=dev,
                    weights_only=False)
    net = ValueQNet(**ck["config"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    print(f"{args.run}: ValueQNet, {sum(p.numel() for p in net.parameters()):,} params")

    # ---- offline: which head names the move the engine actually played? ----
    d = load(paths.data("gomocup15.npz"))
    idx = np.nonzero(d["is_val"])[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, size=min(args.sample, len(idx)), replace=False)
    x = torch.from_numpy(to_planes(d)[idx]).to(dev)
    mv = torch.from_numpy(d["move"][idx].astype(np.int64)).to(dev)
    vt = d["value_target"][idx]
    target = vt[:, 2] - vt[:, 0]

    with torch.no_grad():
        v, p, q = net(x)
    occ_mask = (x[:, 0] + x[:, 1]).reshape(len(idx), -1) > 0.5
    p_masked = p.masked_fill(occ_mask, -1e9)
    q_masked = q.masked_fill(occ_mask, -1e9)
    p_acc = (p_masked.argmax(1) == mv).float().mean().item()
    q_acc = (q_masked.argmax(1) == mv).float().mean().item()
    q_at = q.gather(1, mv.unsqueeze(1)).squeeze(1).cpu().numpy()
    corr = float(np.corrcoef(q_at, target)[0, 1])

    print(f"\noffline, {len(idx):,} held-out positions")
    print(f"  policy head picks the played move : {p_acc:6.1%}")
    print(f"  Q head picks the played move      : {q_acc:6.1%}")
    print(f"  corr(Q at played cell, outcome)   : {corr:+.3f}")

    # ---- on the board ----
    n = args.openings
    openings = make_openings(n)
    ref = frozen_heuristic()
    qa = QAgent(net, dev)
    pa = PolicyAgent2(net, dev)
    s_q = scores(qa, ref, openings)
    s_p = scores(pa, ref, openings)
    print(f"\n{n} openings x 2 colours vs the frozen heuristic")
    print(f"  Q head argmax     : {sum(s_q)/n:6.1%}")
    print(f"  policy head argmax: {sum(s_p)/n:6.1%}")
    d2 = [a - b for a, b in zip(s_q, s_p)]
    m = sum(d2) / len(d2)
    var = sum((v_ - m) ** 2 for v_ in d2) / max(len(d2) - 1, 1)
    se = max(var ** 0.5 / len(d2) ** 0.5, 1e-9)
    print(f"  Q over policy: {m:+.1%} ({m/se:+.1f} sigma, paired)")
    h = scores(qa, pa, openings, seed=777)
    print(f"  head to head, Q vs policy: {sum(h)/n:.1%}")


if __name__ == "__main__":
    main()
