"""Loader for the solver-labelled value dataset.

The .npz stores bitboards split across WORDS uint64 words (4 for 15x15).
Everything a training loop wants -- board planes, class weights, the 8-fold
dihedral augmentation -- is derived here, so the on-disk format stays small and
lossless.
"""
from __future__ import annotations

import paths

import json

import numpy as np

from engine import CELLS, H, W
N_CLASSES = 3


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def bitboards(d):
    """Reassemble the boards as Python ints (one pass, for stats and debug)."""
    return ([unpack_board(r) for r in d["own"]],
            [unpack_board(r) for r in d["opp"]])


# --------------------------------------------------------------------------
# board packing
#
# A 15x15 board needs 225 bits, which does not fit in two uint64 words (128).
# Boards are therefore stored as (N, WORDS) uint64, WORDS = ceil(CELLS / 64).
# Every dataset on disk uses this one layout; nothing else may invent its own.
# --------------------------------------------------------------------------
WORDS = (CELLS + 63) // 64
_W64 = 0xFFFFFFFFFFFFFFFF


def pack_boards(values):
    """[int] -> (N, WORDS) uint64."""
    out = np.zeros((len(values), WORDS), dtype=np.uint64)
    for r, v in enumerate(values):
        for i in range(WORDS):
            out[r, i] = (v >> (64 * i)) & _W64
    return out


def unpack_board(row):
    v = 0
    for i in range(WORDS):
        v |= int(row[i]) << (64 * i)
    return v


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------
def _planes(rows):
    """(N, WORDS) uint64 -> (N, CELLS) uint8."""
    n = rows.shape[0]
    out = np.zeros((n, CELLS), dtype=np.uint8)
    for w in range(WORDS):
        word = rows[:, w]
        base = 64 * w
        for b in range(min(64, CELLS - base)):
            out[:, base + b] = (word >> np.uint64(b)) & np.uint64(1)
    return out


def to_planes(d):
    """(N, 3, H, W) float32: own stones, opponent stones, last move."""
    own = _planes(d["own"])
    opp = _planes(d["opp"])
    n = own.shape[0]
    last = np.zeros((n, CELLS), dtype=np.uint8)
    li = d["last"]
    ok = li >= 0
    last[np.nonzero(ok)[0], li[ok]] = 1
    x = np.stack([own, opp, last], axis=1)
    return x.reshape(n, 3, H, W).astype(np.float32)


# --------------------------------------------------------------------------
# 8-fold dihedral augmentation (value is invariant -> labels unchanged)
# --------------------------------------------------------------------------
def _dihedral_perm(k):
    pr = []
    for r in range(H):
        for c in range(W):
            rr, cc = r, c
            for _ in range(k % 4):
                rr, cc = W - 1 - cc, rr
            if k >= 4:
                rr, cc = cc, rr
            pr.append(rr * W + cc)
    return np.asarray(pr, dtype=np.int64)


DIHEDRAL = [_dihedral_perm(k) for k in range(8)]


def augment(x, k, flat=True):
    """Apply dihedral transform k to a (N,3,H,W) or (N,CELLS) input."""
    perm = DIHEDRAL[k % 8]
    if x.ndim == 4:
        n = x.shape[0]
        return x.reshape(n, 3, -1)[:, :, perm].reshape(x.shape)
    return x[:, perm]


# --------------------------------------------------------------------------
# class weights
# --------------------------------------------------------------------------
def class_weights(cls, n_classes=N_CLASSES):
    """Inverse-frequency weights, normalised to mean 1 over present classes."""
    counts = np.bincount(cls, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    return w, counts


# --------------------------------------------------------------------------
# training-ready split
# --------------------------------------------------------------------------
def prepare(path, augment_k=None):
    d = load(path)
    x = to_planes(d)
    y = d["cls"].astype(np.int64)
    tr = d["is_val"] == 0
    va = ~tr
    cw, counts = class_weights(y[tr])
    if augment_k is not None:
        x = augment(x, augment_k)
    return {
        "x_train": x[tr], "y_train": y[tr],
        "x_val": x[va], "y_val": y[va],
        "class_weight": cw, "class_counts": counts,
        "meta": d["meta"], "ply": d["ply"],
    }


def _data_card(path):
    d = load(path)
    meta = d["meta"]
    y = d["cls"]
    print(f"file            : {path}")
    print(f"board           : {meta.get('board')}")
    print(f"labeller        : VCF depth {meta.get('depth')}")
    print(f"positions       : {len(y):,}")
    print(f"generated from  : {meta.get('raw_positions'):,} raw positions / "
          f"{meta.get('games'):,} games in {meta.get('seconds')}s")
    print()
    w, counts = class_weights(y)
    print(f"{'class':>10} {'count':>9} {'share':>8} {'weight':>8}")
    for c, name in ((0, "loss"), (1, "draw"), (2, "win")):
        print(f"{name:>10} {int(counts[c]):>9,} {counts[c]/len(y):>8.1%} {w[c]:>8.2f}")
    print()
    print(f"+1 : -1 ratio   : {counts[2]/max(counts[0],1):.2f} : 1")
    tr = d["is_val"] == 0
    print(f"train / val     : {tr.sum():,} / {(~tr).sum():,} positions")
    print(f"augmented views : {len(y)*8:,} (8-fold dihedral, built at load time)")
    print()
    ply = d["ply"]
    print("ply distribution of the labelled set:")
    for lo in range(0, 90, 10):
        m = (ply >= lo) & (ply < lo + 10)
        if m.any():
            print(f"  {lo:2d}-{lo+9:2d}: {int(m.sum()):>7,}  "
                  f"win {y[m].mean()/2:.1%}")
    print()
    b = bitboards(d)
    print(f"sample bitboard : own={b[0][0]:#x} opp={b[1][0]:#x} "
          f"cls={int(y[0])} ply={int(ply[0])}")


if __name__ == "__main__":
    import sys
    _data_card(sys.argv[1] if len(sys.argv) > 1 else paths.data("batch1.npz"))
