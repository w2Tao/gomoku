"""Baseline registry: discover, describe and load frozen checkpoints.

Each baseline is a directory under `ckpt/` holding `manifest.json` and `model.pt`.
Nothing here mutates a baseline -- a frozen baseline stays frozen, that is the
point of freezing it.

Two model types exist. `value` is a `ValueNet`, played through a 1-ply search.
`value_policy` is a `ValuePolicyNet`, played straight off its policy head. The
manifest declares which; manifests without the field are treated as `value` so
old ones keep working.
"""
from __future__ import annotations

import json
import os

import torch

import paths
from model import ValueEvalNet, ValueNet, ValuePolicyEvalNet, ValuePolicyNet

BASE_DIR = str(paths.BASELINES)

_CACHE: dict[str, object] = {}


def list_baselines() -> list[dict]:
    out = []
    if not os.path.isdir(BASE_DIR):
        return out
    for name in sorted(os.listdir(BASE_DIR)):
        mpath = os.path.join(BASE_DIR, name, "manifest.json")
        mfile = os.path.join(BASE_DIR, name, "model.pt")
        if not (os.path.isfile(mpath) and os.path.isfile(mfile)):
            continue
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        out.append({
            "id": name,
            "name": m.get("name", name),
            "model_type": m.get("model_type", "value"),
            "params": m.get("parameters"),
            "strength_vs_teacher": m.get("metrics", {}).get("vs_heuristic_paired"),
            "source_step": m.get("source_step"),
            "created": m.get("created"),
            "inference": m.get("inference", {}),
            # which engine the labels came from, if this model was distilled
            "distillation": m.get("distillation"),
        })
    return out


def load_baseline(bid: str, dev: str = "cpu"):
    key = f"{bid}:{dev}"
    if key in _CACHE:
        return _CACHE[key]
    mpath = os.path.join(BASE_DIR, bid, "manifest.json")
    if not os.path.isfile(mpath):
        raise KeyError(f"unknown baseline: {bid}")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    ck = torch.load(os.path.join(BASE_DIR, bid, "model.pt"),
                    map_location=dev, weights_only=False)
    kind = manifest.get("model_type", "value")
    if kind == "value_eval":
        cls = ValueEvalNet
    elif kind == "value_policy_eval":
        cls = ValuePolicyEvalNet
    elif kind == "value_policy":
        cls = ValuePolicyNet
    else:
        cls = ValueNet
    net = cls(**ck["config"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    _CACHE[key] = (net, manifest)
    return net, manifest
