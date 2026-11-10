"""Generate Monte-Carlo value data with a strong rollout policy.

Sampling
--------
Draws from the solver-blind region (mc1) only. The heuristic agrees with the
solver on all 200 held-out positions we checked, so spending rollouts on
solver-decidable positions buys no new information at ~175x the cost of an exact
solver label.

Diversity
---------
`topk` randomises only among the best-scoring *quiet* moves; forced moves (five,
block, VCF attack or defence) stay deterministic. Rollouts differ without ever
blundering tactically.

Parallelism
-----------
`multiprocessing.Pool` cannot be used here: its queues are built on named pipes,
which the sandbox refuses (WinError 5). Instead each shard is a separate OS
process with inherited stdio, writing its results to a file, and the parent
merges them. Sharding is by `index % shards`, so a shard's assignment does not
depend on scheduling and the result is reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

import paths
from dataset import bitboards, load, pack_boards
from engine import CELLS, has_five
from play import HeuristicAgent


def rollout_seq(agent, my: int, their: int, rng):
    """One game, sequential. 0 = mover wins, 1 = opponent wins, None = draw."""
    a, b = my, their
    last = -1
    turn = 0
    for _ in range(CELLS):
        if turn == 0:
            m = agent.move(a, b, last, rng)
            if m is None:
                return None
            a |= 1 << m
            if has_five(a):
                return 0
        else:
            m = agent.move(b, a, last, rng)
            if m is None:
                return None
            b |= 1 << m
            if has_five(b):
                return 1
        last = m
        turn ^= 1
    return None


def build_sources(args):
    """Deterministic list of positions to label, in a fixed order."""
    sources = []
    for name, path, want in (("mc1", paths.data("mc1.npz"), args.from_mc1),
                             ("batch1", paths.data("batch1.npz"),
                              args.from_batch1)):
        if want <= 0:
            continue
        d = load(path)
        own, opp = bitboards(d)
        order = list(range(len(own)))
        random.Random(args.seed + len(sources)).shuffle(order)
        for i in order[:want]:
            sources.append(dict(name=name, idx=i, own=own[i], opp=opp[i],
                                last=int(d["last"][i]),
                                is_val=int(d["is_val"][i]),
                                gid=int(d["game_id"][i]),
                                ply=int(bin(own[i] | opp[i]).count("1"))))
    return sources


def shard_path(k):
    return paths.data(f"_shard_{k}.npz")


def run_shard(args, shard, shards):
    sources = build_sources(args)
    mine = [(k, s) for k, s in enumerate(sources) if k % shards == shard]
    agent = HeuristicAgent(vcf_depth=7, use_vcf=True, use_score=True,
                           topk=args.topk)
    rng = random.Random(args.seed * 100003 + shard)
    counts = np.zeros((len(mine), 3), dtype=np.uint16)
    t0 = time.time()
    for j, (k, s) in enumerate(mine):
        w = dr = 0
        for _ in range(args.rollouts):
            r = rollout_seq(agent, s["own"], s["opp"], rng)
            if r is None:
                dr += 1
            elif r == 0:
                w += 1
        counts[j] = (args.rollouts - w - dr, dr, w)
    np.savez(shard_path(shard),
             idx=np.array([k for k, _ in mine], dtype=np.int32),
             counts=counts)
    print(f"  shard {shard:2d}: {len(mine):5d} positions in "
          f"{time.time()-t0:5.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--from-mc1", type=int, default=5000)
    ap.add_argument("--from-batch1", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="mc4.npz")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    if not os.path.dirname(args.out):      # bare name -> land in data/
        args.out = paths.data(args.out)

    if args.agent != "heuristic":
        raise SystemExit("only the heuristic is supported in this parallel path")
    if args.shard >= 0:
        run_shard(args, args.shard, args.shards)
        return

    workers = args.workers or max(1, min(12, (os.cpu_count() or 4) - 1))
    sources = build_sources(args)
    n = len(sources)
    m1 = sum(1 for s in sources if s["name"] == "mc1")
    print(f"rollout policy : heuristic (VCF 7 + score, topk={args.topk})")
    print(f"positions      : {n}  (mc1 {m1}, batch1 {n-m1})")
    print(f"rollouts each  : {args.rollouts}")
    print(f"workers        : {workers}")

    for k in range(workers):
        sp = shard_path(k)
        if os.path.isfile(sp):
            os.remove(sp)

    passthrough = ["--agent", args.agent, "--from-mc1", str(args.from_mc1),
                   "--from-batch1", str(args.from_batch1),
                   "--rollouts", str(args.rollouts), "--topk", str(args.topk),
                   "--seed", str(args.seed), "--shards", str(workers)]
    t0 = time.time()
    procs = []
    for k in range(workers):
        cmd = [sys.executable, os.path.abspath(__file__)] + passthrough + \
              ["--shard", str(k)]
        procs.append(subprocess.Popen(cmd, stdout=None, stderr=None))
    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        raise SystemExit(f"a shard failed: {codes}")

    counts = np.zeros((n, 3), dtype=np.uint16)
    filled = np.zeros(n, dtype=bool)
    for k in range(workers):
        sp = shard_path(k)
        if not os.path.isfile(sp):
            raise SystemExit(f"missing shard file {sp}")
        z = np.load(sp)
        try:
            counts[z["idx"]] = z["counts"]
            filled[z["idx"]] = True
        finally:
            z.close()          # npz keeps the handle open; Windows needs it shut
        os.remove(sp)
    if not filled.all():
        raise SystemExit(f"{int((~filled).sum())} positions were never labelled")
    dt = time.time() - t0

    out = {
        "own": pack_boards([s["own"] for s in sources]),
        "opp": pack_boards([s["opp"] for s in sources]),
        "last": np.array([s["last"] for s in sources], dtype=np.int16),
        "ply": np.array([s["ply"] for s in sources], dtype=np.uint8),
        "is_val": np.array([s["is_val"] for s in sources], dtype=np.uint8),
        "game_id": np.array([s["gid"] for s in sources], dtype=np.uint32),
        "src_name": np.array([0 if s["name"] == "mc1" else 1 for s in sources],
                             dtype=np.uint8),
        "src_index": np.array([s["idx"] for s in sources], dtype=np.uint32),
        "counts": counts,
        "meta": np.array(json.dumps({
            "rollout_policy": f"heuristic (VCF 7 + score, topk={args.topk})",
            "rollouts": args.rollouts, "positions": int(n),
            "from_mc1": m1, "from_batch1": n - m1,
            "workers": workers, "seconds": round(dt, 1),
        })),
    }
    p = paths.data(args.out)
    np.savez_compressed(p, **out)

    prob = counts.astype(np.float32) / counts.sum(1, keepdims=True)
    v = prob[:, 2] - prob[:, 0]
    print()
    print(f"time    : {dt:.0f}s  ({n/dt:.2f} pos/s, "
          f"{n*args.rollouts/dt:.0f} games/s)")
    print(f"mean V  : {v.mean():+.3f}   std V: {v.std():.3f}   "
          f"|V|>0.5: {(np.abs(v) > 0.5).mean():.1%}")
    print(f"saved   : {p}")


if __name__ == "__main__":
    main()
