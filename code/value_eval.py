"""Measure the value head on held-out Gomocup positions.

The v2 search retry rests on one premise: the value head of the larger base model
is less coarse than the one that made 2-ply search lose in report.md section 7.7.
This script checks that premise directly -- three-class accuracy, balanced
accuracy, and MAE of V = P(win) - P(loss) -- on the same held-out games for every
model.

    python code/value_eval.py baseline,v1,v2s1,v2base
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "code")

import baselines                                       # noqa: E402
import paths                                           # noqa: E402
from dataset import load, to_planes                    # noqa: E402

BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 45), (46, 999)]


def main():
    models = (sys.argv[1] if len(sys.argv) > 1
              else "baseline,v1,v2s1,v2base").split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load(paths.data("gomocup15.npz"))
    sel = np.nonzero(d["is_val"])[0]
    x = torch.from_numpy(to_planes(d)[sel]).to(dev)
    y = torch.from_numpy(d["cls"][sel].astype(np.int64)).to(dev)
    v_true = (y == 2).float() - (y == 0).float()
    ply = d["ply"][sel].astype(np.int64)

    print(f"{len(sel):,} held-out positions from "
          f"{len(np.unique(d['game_id'][sel])):,} games\n")
    print(f"{'model':<10}{'acc':>8}{'bal':>8}{'MAE':>8}{'corr':>8}   by ply")
    print("-" * 74)

    for mid in models:
        net, _ = baselines.load_baseline(mid, dev)
        preds = np.empty(len(sel), dtype=np.int64)
        probs_all = np.empty((len(sel), 3), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(sel), 512):
                out = net(x[i:i + 512])
                logits = out[0] if isinstance(out, tuple) else out
                p = torch.softmax(logits, 1)
                probs_all[i:i + 512] = p.cpu().numpy()
                preds[i:i + 512] = p.argmax(1).cpu().numpy()
        acc = float((preds == y.cpu().numpy()).mean())
        rec = []
        for c in range(3):
            m = y.cpu().numpy() == c
            if m.any():
                rec.append(float((preds[m] == c).mean()))
        bal = float(np.mean(rec)) if rec else 0.0
        v_pred = probs_all[:, 2] - probs_all[:, 0]
        mae = float(np.abs(v_pred - v_true.cpu().numpy()).mean())
        corr = float(np.corrcoef(v_pred, v_true.cpu().numpy())[0, 1])
        cells = []
        for a, b in BUCKETS:
            m = (ply >= a) & (ply <= b)
            if m.any():
                cells.append(f"{a}-{b if b < 999 else '+'}:"
                             f"{float((preds[m] == y.cpu().numpy()[m]).mean()):.1%}")
        print(f"{mid:<10}{acc:>8.1%}{bal:>8.1%}{mae:>8.3f}{corr:>+8.3f}   "
              + "  ".join(cells))


if __name__ == "__main__":
    main()
