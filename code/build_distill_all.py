"""Combine the RAPFI principal-variation collections into one distilled dataset.

The v1 model was trained on a single collection (`dist3`: 49,849 labels). For a
larger model we can afford more data, so this merges dist2 (opening 2), dist3
(opening 6) and dist6 (1,000 ms teacher) and collapses exact duplicate
(position, move) rows. The three collections were built from independent game
batches and overlap by about a quarter of their rows; deduplicating matters for
the sampling distribution even though every copy carries the same label.

Value supervision is deliberately absent: these datasets contain policy labels
only (`value_weight` is 0). The intended recipe is to train a value+policy base
on `gomocup15.npz` first, then distil this policy on top of it, so the value head
keeps the base model's outcome supervision.

    python code/build_distill_all.py --out dist_all.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import paths

FIELDS = ("own", "opp", "last", "ply", "cls", "game_id", "move",
          "value_target", "value_weight")


def load_part(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files if k != "meta"}, json.loads(str(d["meta"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="dist2,dist3,dist6",
                    help="comma-separated names under data/ (or paths)")
    ap.add_argument("--out", default="dist_all.npz")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out if os.path.dirname(args.out) else paths.data(args.out)
    names = [s.strip() for s in args.inputs.split(",") if s.strip()]

    parts, sources = [], []
    for name in names:
        p = name if os.path.isfile(name) else paths.data(
            name if name.endswith(".npz") else name + ".npz")
        d, meta = load_part(p)
        parts.append(d)
        sources.append(meta.get("source", p))
        print(f"  {os.path.basename(p):<16} {len(d['cls']):>7,} rows  "
              f"{meta.get('source', '')}")

    raw = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    n_raw = len(raw["cls"])

    # offset game ids so the by-game split cannot collide across collections
    chunks, off = [], 0
    for p in parts:
        g = p["game_id"].astype(np.int64)
        chunks.append(g + off)
        off += int(g.max()) + 1 if len(g) else 0
    game_id = np.concatenate(chunks)

    # exact duplicate positions+moves: keep the first copy
    own = raw["own"]
    opp = raw["opp"]
    last = raw["last"].astype(np.int64)
    move = raw["move"].astype(np.int64)
    seen, keep = set(), []
    for i in range(n_raw):
        key = (own[i].tobytes(), opp[i].tobytes(), int(last[i]), int(move[i]))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    kept = {k: raw[k][keep] for k in FIELDS}
    kept["game_id"] = game_id[keep]
    print(f"  dedupe          {n_raw:,} -> {len(keep):,} rows "
          f"({n_raw - len(keep):,} duplicates)")

    gid = kept["game_id"]
    uniq = np.unique(gid)
    rng = np.random.default_rng(args.seed)
    val_games = set(rng.choice(
        uniq, size=max(1, int(len(uniq) * args.val_frac)), replace=False).tolist())
    kept["is_val"] = np.isin(gid, list(val_games))
    kept["meta"] = np.array(json.dumps({
        "source": "RAPFI25 PV distillation, merged collections",
        "sources": sources,
        "inputs": names,
        "positions": int(len(kept["cls"])),
        "raw_positions": int(n_raw),
        "games": int(len(uniq)),
        "split": "by game",
        "labels": "policy only from RAPFI25 principal variations; "
                  "value_weight is zero, cls is a placeholder",
    }))

    np.savez_compressed(out, **kept)
    print(f"\nsaved -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"positions {len(kept['cls']):,} | games {len(uniq):,} | "
          f"val rows {int(kept['is_val'].sum()):,}")


if __name__ == "__main__":
    main()
