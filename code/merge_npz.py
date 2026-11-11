"""Concatenate shard datasets into one, keeping the by-game validation split.

Shards are written independently by parallel workers, so game ids restart at 0 in
each. They are renumbered with an offset -- using the self-play shard's own maximum
instead of the base one silently collided ids and leaked the split, so the offset
is taken from the running total here.
"""
from __future__ import annotations

import os
import sys

import json

import numpy as np

sys.path.insert(0, "code")

import paths                                             # noqa: E402
from dataset import load                                 # noqa: E402

FIELDS = ("own", "opp", "last", "ply", "cls", "game_id", "move",
          "value_target", "value_weight")


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: merge_npz.py OUT.npz A.npz B.npz ...")
    out_path = sys.argv[1]
    if not os.path.isabs(out_path) and os.path.dirname(out_path) == "":
        out_path = paths.data(out_path)      # bare name means "in data/"
    parts = [load(p if p.endswith(".npz") else paths.data(p))
             for p in sys.argv[2:]]

    total, offset, merged = 0, 0, {}
    for p in parts:
        n = len(p["cls"])
        for k in FIELDS:
            v = p[k]
            if k == "game_id":
                v = v + offset
            merged.setdefault(k, []).append(v)
        total += n
        offset = int(merged["game_id"][-1].max()) + 1 if merged["game_id"] else 0

    for k in FIELDS:
        merged[k] = np.concatenate(merged[k])

    games = np.unique(merged["game_id"])
    rng = np.random.default_rng(0)
    val = set(rng.choice(games, size=max(1, int(len(games) * 0.1)),
                         replace=False).tolist())
    merged["is_val"] = np.isin(merged["game_id"], list(val))
    # dataset.load requires meta
    merged["meta"] = np.array(json.dumps({
        "source": f"merged from {len(parts)} shards: "
                  + ", ".join(os.path.basename(p) for p in sys.argv[2:]),
        "positions": int(total),
        "games": int(len(games)),
        "split": "by game",
    }))

    np.savez_compressed(out_path, **merged)
    print(f"merged {len(parts)} shards -> {out_path}")
    print(f"  positions {total:,} | games {len(games):,} | "
          f"val {int(merged['is_val'].sum()):,}")


if __name__ == "__main__":
    main()
