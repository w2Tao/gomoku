"""Freeze a training checkpoint as an immutable named baseline.

A baseline is only useful if it is reproducible: the weights, the architecture,
the inference recipe and the measured strength all have to travel together.

    python code/freeze_baseline.py --src t2_finetune --id baseline2
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
    ap.add_argument("--id", default="baseline2")
    ap.add_argument("--name", default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--vs-greedy", type=float, default=None,
                    help="measured paired score against the frozen heuristic, if known")
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

    # A `ValuePolicyNet` carries a policy head; play it straight off that head.
    has_policy = any(k.startswith("policy_") for k in ck["state_dict"])
    model_type = "value_policy" if has_policy else "value"
    n_par = metrics.get("params") or sum(
        v.numel() for v in ck["state_dict"].values())

    if has_policy:
        recipe = {
            "head": "policy",
            "search": "none",
            "selection": "argmax over legal cells, deterministic",
        }
        limits = [
            "one forward pass: no VCF, no double-three, no search",
            "imitation of its label source, so it cannot exceed that source",
        ]
    else:
        recipe = {
            "head": "value",
            "search_depth_plies": args.depth,
            "selection": "argmax, deterministic",
        }
        limits = [
            "value is a calibrated-ish score, not a probability",
            "2-ply search is a net loss at this value accuracy",
        ]

    manifest = {
        "id": args.id,
        "name": args.name or f"{args.id} · {model_type}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_type": model_type,
        "source_run": args.src,
        "source_checkpoint": src_ckpt,
        "source_step": ck.get("step"),
        "board": "15x15 freestyle gomoku, no forbidden moves",
        "architecture": ck["config"],
        "parameters": n_par,
        "training": {
            "data": margs.get("data") or info.get("data"),
            "steps": margs.get("steps"),
            "augmentation": "8-fold dihedral, per sample",
        },
        "metrics": {
            "val_loss": best.get("loss"),
            "vs_heuristic_paired": args.vs_greedy,
        },
        "inference": recipe,
        "known_limits": limits,
    }
    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"frozen  : ckpt/{args.id}/model.pt "
          f"({os.path.getsize(dst + '/model.pt')/1e6:.2f} MB)")
    print(f"source  : runs/{args.src} @ step {manifest['source_step']}")
    print(f"params  : {manifest['parameters']:,}")
    print(f"type    : {model_type}")
    if best.get("loss") is not None:
        print(f"val loss: {best.get('loss'):.4f}")
    if args.vs_greedy is not None:
        print(f"strength: {args.vs_greedy:.1%} vs the frozen heuristic")


if __name__ == "__main__":
    main()
