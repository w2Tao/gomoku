"""Merge eval-dataset shards and rebuild the by-game validation split."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "code")

import paths                                             # noqa: E402

FIELDS = ("own", "opp", "last", "ply", "eval", "depth", "game_id")


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: merge_eval_shards.py OUT.npz A.npz B.npz ...")
    out = sys.argv[1]
    parts = [np.load(p if os.path.isabs(p) or os.path.dirname(p)
                     else paths.data(p), allow_pickle=False)
             for p in sys.argv[2:]]
    merged = {}
    for p in parts:
        for k in FIELDS:
            merged.setdefault(k, []).append(p[k])
    for k in FIELDS:
        merged[k] = np.concatenate(merged[k])
    gid = merged["game_id"].astype(np.int64)
    # Shards sample disjoint positions from the same original games, so the
    # game_id is already global. Do not offset it per shard: that would turn one
    # game into several ids and leak the by-game validation split.
    merged["game_id"] = gid
    games = np.unique(gid)
    rng = np.random.default_rng(0)
    val = set(rng.choice(games, size=max(1, int(len(games) * 0.1)),
                         replace=False).tolist())
    merged["is_val"] = np.isin(gid, list(val))
    merged["eval_weight"] = np.ones(len(gid), dtype=np.float32)
    merged["meta"] = np.array(json.dumps({
        "source": "RAPFI25 Eval, merged shards",
        "positions": int(len(gid)),
        "games": int(len(games)),
        "split": "by game",
        "eval_scale": 200.0,
    }))
    out = out if os.path.dirname(out) else paths.data(out)
    np.savez_compressed(out, **merged)
    print(f"merged {len(parts)} shards -> {out}")
    print(f"  positions {len(gid):,} | games {len(games):,} | "
          f"val {int(merged['is_val'].sum()):,}")


if __name__ == "__main__":
    main()
