"""Freeze a training checkpoint as an immutable named baseline.

A baseline is only useful if it is reproducible: the weights, the architecture,
the inference recipe and the measured strength all have to travel together.

    python code/freeze_baseline.py --src v3 --id baseline1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="v3", help="run tag under ckpt/runs/")
    ap.add_argument("--id", default="baseline1")
    ap.add_argument("--name", default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--vs-greedy", type=float, default=None,
                    help="measured paired win rate against greedy+, if known")
    args = ap.parse_args()

    src_ckpt = paths.run_ckpt(args.src)
    metrics_path = paths.run_ckpt(args.src, "metrics.json")
    if not os.path.isfile(src_ckpt):
        raise SystemExit(f"no checkpoint at {src_ckpt}")

    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    hist = metrics.get("history") or [{}]
    best = min(hist, key=lambda h: h.get("loss", 1e9))
    info = metrics.get("info", {})
    margs = metrics.get("args", {})

    dst = paths.baseline_dir(args.id)
    os.makedirs(dst, exist_ok=True)
    torch.save(ck, os.path.join(dst, "model.pt"))

    manifest = {
        "id": args.id,
        "name": args.name or f"{args.id} · value net + {args.depth}-ply search",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_run": args.src,
        "source_checkpoint": src_ckpt,
        "source_step": ck.get("step"),
        "board": "9x9 freestyle gomoku, no forbidden moves",
        "architecture": ck["config"],
        "parameters": metrics.get("params"),
        "training": {
            "solver_labels": info.get("solver"),
            "rollout_labels": info.get("mc"),
            "augmentation": "8-fold dihedral, per sample",
            "steps": margs.get("steps"),
            "rollout_weight": margs.get("mc_weight"),
        },
        "metrics": {
            "solver_balanced_accuracy": best.get("solver_bal"),
            "rollout_mae": best.get("mc_mae"),
            "rollout_corr": best.get("mc_corr"),
            "val_loss": best.get("loss"),
            "vs_greedy_plus_paired": args.vs_greedy,
        },
        "inference": {
            "search_depth_plies": args.depth,
            "selection": "argmax, deterministic",
            "opening_randomisation": "only for evaluation, not in play",
        },
        "known_limits": [
            "value is a calibrated-ish score, not a probability",
            "rollout labels are V^pi of the heuristic player, not optimal-play value",
            "2-ply search is a net loss at this value accuracy",
        ],
    }
    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"frozen  : ckpt/{args.id}/model.pt "
          f"({os.path.getsize(dst + '/model.pt')/1e6:.2f} MB)")
    print(f"source  : runs/{args.src} @ step {manifest['source_step']}")
    print(f"params  : {manifest['parameters']:,}")
    print(f"quality : solver bal {best.get('solver_bal')}, "
          f"rollout MAE {best.get('mc_mae')}, corr {best.get('mc_corr')}")


if __name__ == "__main__":
    main()
