"""Calibrate RAPFI's `Eval` scale against actual game outcomes.

`train_policy.py --eval-head` maps Eval to tanh(Eval / T). T is not universal: it
depends on the engine's score units and on how conservative its search is. This
script scans T on the eval-labelled positions collected by
`build_distill_dataset.py` and reports the logistic loss / sign accuracy against
the eventual game result.

    python code/eval_calibrate.py --data dist_eval.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dist_eval.npz")
    ap.add_argument("--temps", default="50,75,100,150,200,300,400,500,700,1000,1500,2000,3000")
    ap.add_argument("--max-eval", type=int, default=100000)
    args = ap.parse_args()
    path = args.data if os.path.isfile(args.data) else paths.data(args.data)
    d = np.load(path, allow_pickle=False)
    ev = d["eval"].astype(np.int64)
    ew = d["eval_weight"].astype(np.float32)
    out = d["outcome"].astype(np.int64)
    mask = (ew > 0) & (out >= 0) & (np.abs(ev) <= args.max_eval)
    x = ev[mask].astype(np.float64)
    y = out[mask].astype(np.float64)
    print(f"{path}: {len(x):,} eval labels with a decided game")
    if not len(x):
        raise SystemExit("no usable eval labels")
    print(f"  eval range {int(x.min())} .. {int(x.max())}  "
          f"mean {x.mean():+.1f}  median {np.median(x):+.1f}")
    print(f"  mover win rate {y.mean():.1%}  "
          f"sign(eval) == outcome {((x > 0) == (y > 0.5)).mean():.1%}")
    print()
    print(f"{'T':>8}{'logloss':>10}{'brier':>8}{'acc':>8}")
    best = None
    for t in [float(s) for s in args.temps.split(",") if s.strip()]:
        p = 1.0 / (1.0 + np.exp(-np.clip(x / t, -30, 30)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logloss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        brier = float(((p - y) ** 2).mean())
        acc = float(((p > 0.5) == (y > 0.5)).mean())
        print(f"{t:>8.0f}{logloss:>10.4f}{brier:>8.4f}{acc:>8.1%}")
        if best is None or logloss < best[1]:
            best = (t, logloss, brier, acc)
    print()
    print(f"best T = {best[0]:.0f}  logloss {best[1]:.4f}  "
          f"brier {best[2]:.4f}  acc {best[3]:.1%}")


if __name__ == "__main__":
    main()
