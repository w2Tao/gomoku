"""Freeze a training checkpoint as an immutable named baseline.

A baseline is only useful if it is reproducible: the weights, the architecture,
the inference recipe and the measured strength all have to travel together.

    python code/freeze_baseline.py --src b2_fixed --id baseline --vs-heuristic 0.630
    python code/freeze_baseline.py --src d3v2 --id v1 --vs-heuristic 0.757 \
        --teacher RAPFI25 --opening 6

The optional `--vs-heuristic` value is documented with its measurement in
`report.md`; the manifest records that note next to the number so the strength
claim never travels without its provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

import paths


def _history_loss(entry, lam, eval_weight=0.0):
    """The objective actually selected on during training.

    `train.py` records a single `loss`; `train_policy.py` records `value_loss` and
    `policy_loss` separately, and `ValuePolicyEvalNet` runs add `eval_loss`.
    Reading only `loss` (the old behaviour) silently produced `val_loss: null`
    for every value_policy baseline.
    """
    if entry.get("loss") is not None:
        return float(entry["loss"])
    loss = (float(entry.get("value_loss", 0.0))
            + lam * float(entry.get("policy_loss", 0.0)))
    if eval_weight:
        loss += eval_weight * float(entry.get("eval_loss", 0.0))
    return loss


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="v3", help="run tag under ckpt/runs/")
    ap.add_argument("--id", default="baseline")
    ap.add_argument("--name", default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--vs-heuristic", "--vs-greedy", dest="vs_heuristic",
                    type=float, default=None,
                    help="measured paired score against the frozen heuristic")
    ap.add_argument("--metric-note", default=None,
                    help="provenance for --vs-heuristic (report section, sample size)")
    ap.add_argument("--teacher", default=None,
                    help="distillation teacher, e.g. 'RAPFI25 principal variations'")
    ap.add_argument("--opening", type=int, default=None,
                    help="random opening plies used when the teacher was collected")
    ap.add_argument("--distill-note", default=None)
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
    hist = metrics.get("history") or []
    info = metrics.get("info", {})
    margs = metrics.get("args", {})
    lam = float(margs.get("lam", 1.0))
    eval_weight = (float(margs.get("eval_weight", 1.0))
                   if margs.get("eval_head") else 0.0)

    if hist:
        best = min(hist, key=lambda h: _history_loss(h, lam, eval_weight))
        best_loss = _history_loss(best, lam, eval_weight)
    else:
        best, best_loss = {}, None

    dst = paths.baseline_dir(args.id)
    os.makedirs(dst, exist_ok=True)
    model_path = os.path.join(dst, "model.pt")
    torch.save(ck, model_path)

    # A `ValuePolicyNet` carries a policy head; play it straight off that head.
    has_policy = any(k.startswith("policy_") for k in ck["state_dict"])
    has_eval = any(k.startswith("eval_") for k in ck["state_dict"])
    if has_eval:
        model_type = "value_policy_eval"
    elif has_policy:
        model_type = "value_policy"
    else:
        model_type = "value"
    n_par = metrics.get("params") or sum(
        v.numel() for v in ck["state_dict"].values())

    if has_eval:
        recipe = {
            "head": "policy + value (eval-trained scalar)",
            "search": "policy prior; scalar value for leaves",
            "selection": "policy argmax at the root",
        }
        limits = [
            "the scalar head imitates RAPFI's Eval, not game-theoretic values",
            "search over this value has not been shown to beat the policy head",
        ]
    elif has_policy:
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

    data_path = margs.get("data") or info.get("data")
    if data_path and not os.path.isfile(data_path):
        limits.append(
            f"training data {data_path} is not present in this checkout; "
            "the weights are frozen but cannot be retrained here")

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
        "sha256": _sha256(model_path),
        "training": {
            "data": data_path,
            "steps": margs.get("steps"),
            "warm_start": margs.get("warm") or None,
            "learning_rate": margs.get("lr"),
            "batch": margs.get("batch"),
            "seed": margs.get("seed"),
            "eval_head": bool(margs.get("eval_head")),
            "eval_weight": margs.get("eval_weight"),
            "eval_temp": margs.get("eval_temp"),
            "augmentation": "8-fold dihedral, per sample",
        },
        "metrics": {
            "val_loss": best_loss,
            "value_loss": best.get("value_loss"),
            "policy_loss": best.get("policy_loss"),
            "policy_acc": best.get("policy_acc"),
            "eval_loss": best.get("eval_loss"),
            "eval_mae": best.get("eval_mae"),
            "eval_labels": best.get("eval_labels"),
            "eval_weight": eval_weight,
            "step": best.get("step"),
            "loss_kind": ("value + policy + eval" if has_eval
                          else ("value + policy" if has_policy else "value")),
            "vs_heuristic_paired": args.vs_heuristic,
            "vs_heuristic_note": args.metric_note,
        },
        "inference": recipe,
        "known_limits": limits,
    }
    if args.teacher:
        manifest["distillation"] = {
            "teacher": args.teacher,
            "opening_plies": args.opening,
            "data": data_path,
            "note": args.distill_note,
        }
    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"frozen  : ckpt/{args.id}/model.pt "
          f"({os.path.getsize(model_path)/1e6:.2f} MB)")
    print(f"source  : runs/{args.src} @ step {manifest['source_step']}")
    print(f"params  : {manifest['parameters']:,}")
    print(f"type    : {model_type}")
    print(f"sha256  : {manifest['sha256']}")
    if best_loss is not None:
        print(f"val loss: {best_loss:.4f} ({manifest['metrics']['loss_kind']})")
    if args.vs_heuristic is not None:
        print(f"strength: {args.vs_heuristic:.1%} vs the frozen heuristic")


if __name__ == "__main__":
    main()
