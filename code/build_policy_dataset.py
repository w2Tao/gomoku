"""Distil the 1-ply agent into a policy target.

Why distillation rather than solver labels
------------------------------------------
A policy label can be derived from the solver exactly ("which moves still
preserve a proven win"), but that proof costs ~45 solver calls per position and
only exists where the solver can already prove a win -- roughly the late game.
A rollout policy needs to be sane *everywhere*, including the opening, where the
solver proves nothing at all.

So the teacher here is the current agent: value network + 1-ply search, measured
at 85.2% against greedy. The policy head learns to reproduce its choice in a
single forward pass. It cannot be stronger than the teacher; it is ~45x cheaper,
which is exactly what a rollout policy needs.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

import baselines
import paths
from dataset import bitboards, class_weights, load, pack_boards
from play import HeuristicAgent, SearchAgent

MC_WEIGHT = 0.5


def collect(path, kind):
    d = load(path)
    own, opp = bitboards(d)
    n = len(own)
    if kind == "solver":
        cls = d["cls"].astype(np.int64)
        t = np.zeros((n, 3), np.float32)
        t[np.arange(n), cls] = 1.0
        cw, _ = class_weights(cls)
        w = cw[cls].astype(np.float32)
    else:
        c = d["counts"].astype(np.float32)
        t = (c / c.sum(1, keepdims=True)).astype(np.float32)
        w = np.full(n, MC_WEIGHT, np.float32)
    return dict(own=own, opp=opp, last=d["last"].astype(np.int16),
                ply=d["ply"].astype(np.uint8), gid=d["game_id"].astype(np.uint32),
                val=d["is_val"].astype(np.uint8), t=t, w=w, n=n)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="heuristic",
                    choices=["heuristic", "search"],
                    help="heuristic = VCF+score (strongest); search = baseline0 + 1-ply")
    ap.add_argument("--out", default="policy1.npz")
    args = ap.parse_args()
    if not os.path.dirname(args.out):      # bare name -> land in data/
        args.out = paths.data(args.out)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.teacher == "heuristic":
        agent = HeuristicAgent(vcf_depth=7, use_vcf=True, use_score=True)
        teacher_desc = "heuristic (VCF depth 7 + static threat score)"
    else:
        net, _man = baselines.load_baseline("baseline0", dev)
        agent = SearchAgent(net, dev, plies=1)
        teacher_desc = "baseline0 (value net + 1-ply search)"
    print(f"teacher : {teacher_desc}, device {dev}")

    parts = [collect(paths.data("batch1.npz"), "solver"),
             collect(paths.data("mc1.npz"), "mc")]
    n = sum(p["n"] for p in parts)
    print(f"positions: {n:,}  (solver {parts[0]['n']:,} + mc {parts[1]['n']:,})")

    own = np.concatenate([p["own"] for p in parts])
    opp = np.concatenate([p["opp"] for p in parts])
    last = np.concatenate([p["last"] for p in parts])
    ply = np.concatenate([p["ply"] for p in parts])
    gid = np.concatenate([p["gid"] for p in parts])
    val = np.concatenate([p["val"] for p in parts])
    vt = np.concatenate([p["t"] for p in parts])
    vw = np.concatenate([p["w"] for p in parts])

    rng = random.Random(0)
    moves = np.empty(n, dtype=np.uint8)
    t0 = time.time()
    for i in range(n):
        m = agent.move(int(own[i]), int(opp[i]), int(last[i]), rng)
        moves[i] = 255 if m is None else m
        if (i + 1) % 10000 == 0:
            dt = time.time() - t0
            print(f"  {i+1:6d}/{n}  ({i/dt:5.0f}/s, {dt:5.0f}s)", flush=True)
    dt = time.time() - t0

    out = {
        "own": pack_boards(own),
        "opp": pack_boards(opp),
        "last": last, "ply": ply, "game_id": gid, "is_val": val,
        "value_target": vt, "value_weight": vw, "move": moves,
        "meta": np.array(json.dumps({
            "teacher": teacher_desc,
            "teacher_strength_vs_greedy": 0.815 if args.teacher == "heuristic" else 0.852,
            "positions": int(n),
            "source": ["batch1.npz", "mc1.npz"],
            "seconds": round(dt, 1),
        })),
    }
    os_path = paths.data(args.out)
    np.savez_compressed(os_path, **out)
    print()
    print(f"labelled  : {n:,} positions in {dt:.0f}s")
    print(f"targets   : policy move = argmax of the 1-ply Q")
    print(f"saved     : {os_path}")


if __name__ == "__main__":
    main()
