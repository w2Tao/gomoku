"""Evaluate a ValuePolicyEvalNet's scalar head on held-out RAPFI eval labels.

Two questions:
  1. does the scalar head reproduce tanh(Eval/T)? (MAE against the teacher)
  2. does the sign/soft value predict the actual game result? (BCE / sign accuracy)

The second one is what search needs; the first can be perfect while still being
useless if the teacher's eval is itself poorly calibrated.

    python code/eval_scalar_eval.py v3eval --temp 400
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "code")

import baselines                                       # noqa: E402
import paths                                           # noqa: E402
from dataset import load, to_planes                    # noqa: E402

BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 45), (46, 999)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="v3eval")
    ap.add_argument("--data", default="dist_eval.npz")
    ap.add_argument("--temp", type=float, default=400.0)
    args = ap.parse_args()
    path = args.data if os.path.isfile(args.data) else paths.data(args.data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load(path)
    sel = np.nonzero((d["is_val"]) & (d["eval_weight"] > 0)
                     & (d["outcome"] >= 0))[0]
    if not len(sel):
        raise SystemExit("no held-out eval labels with a decided game")
    x = torch.from_numpy(to_planes(d)[sel]).to(dev)
    ev = d["eval"][sel].astype(np.float32)
    outcome = d["outcome"][sel].astype(np.float32)
    ply = d["ply"][sel].astype(np.int64)
    target = np.tanh(ev / args.temp)

    net, man = baselines.load_baseline(args.model, dev)
    if man.get("model_type") != "value_policy_eval":
        raise SystemExit(f"{args.model} has no eval head")
    out = []
    with torch.no_grad():
        for i in range(0, len(sel), 512):
            o = net(x[i:i + 512])
            out.append(o[2].cpu().numpy())
    scalar = np.concatenate(out)
    mae = float(np.abs(scalar - target).mean())
    corr = float(np.corrcoef(scalar, target)[0, 1])
    p = np.clip((scalar + 1.0) / 2.0, 1e-6, 1 - 1e-6)
    bce = float(-(outcome * np.log(p) + (1 - outcome) * np.log(1 - p)).mean())
    acc = float(((scalar > 0) == (outcome > 0.5)).mean())
    # raw teacher eval at the same temperature, as a calibration reference
    p_ev = np.clip(1.0 / (1.0 + np.exp(-np.clip(ev / args.temp, -30, 30))),
                   1e-6, 1 - 1e-6)
    bce_ev = float(-(outcome * np.log(p_ev)
                     + (1 - outcome) * np.log(1 - p_ev)).mean())
    acc_ev = float(((ev > 0) == (outcome > 0.5)).mean())

    print(f"{args.model} on {len(sel):,} held-out eval labels, T={args.temp:g}")
    print(f"  scalar vs tanh(Eval/T): MAE {mae:.4f}  corr {corr:+.4f}")
    print(f"  scalar vs outcome     : BCE {bce:.4f}  sign acc {acc:.1%}")
    print(f"  raw Eval vs outcome   : BCE {bce_ev:.4f}  sign acc {acc_ev:.1%}")
    print("\n  by ply:")
    for a, b in BUCKETS:
        m = (ply >= a) & (ply <= b)
        if not m.any():
            continue
        print(f"    {a:>2}-{b if b < 999 else '+':<2}: n {int(m.sum()):>5,}  "
              f"MAE {float(np.abs(scalar[m] - target[m]).mean()):.4f}  "
              f"BCE {float(-(outcome[m] * np.log(p[m]) + (1 - outcome[m]) * np.log(1 - p[m])).mean()):.4f}")


if __name__ == "__main__":
    main()
