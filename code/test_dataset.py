"""Tests for the dataset format and loader.

The dangerous failure mode here is silent: a wrong bit unpacking or a botched
dihedral permutation still produces well-shaped tensors and a training run that
converges to something plausible-looking. So every transform is checked against
an independent path (the Python-int bitboards, or the engine's own predicates).
"""

import os

import paths
import numpy as np

from dataset import (
    DIHEDRAL,
    augment,
    bitboards,
    class_weights,
    load,
    to_planes,
)
from engine import CELLS, H, W, has_five
from value_target import DIHEDRAL8, apply_perm


def main():
    if not os.path.isfile(paths.data("_smoke.npz")):
        print("SKIP: data/_smoke.npz missing -- regenerate a dataset first "
              "(python code/build_dataset.py --target 3000 --out _smoke.npz)")
        return
    d = load(paths.data("_smoke.npz"))
    n = len(d["cls"])
    assert n > 0

    # --- bitboards round-trip -------------------------------------------
    own, opp = bitboards(d)
    x = to_planes(d)
    assert x.shape == (n, 3, H, W), x.shape
    for i in range(0, n, 97):
        plane = (x[i, 0] > 0.5).reshape(-1)
        rebuilt = 0
        for c in range(CELLS):
            if plane[c]:
                rebuilt |= 1 << c
        assert rebuilt == own[i], f"own bitboard mismatch at {i}"
        plane = (x[i, 1] > 0.5).reshape(-1)
        rebuilt = 0
        for c in range(CELLS):
            if plane[c]:
                rebuilt |= 1 << c
        assert rebuilt == opp[i], f"opp bitboard mismatch at {i}"
        # own and opp never overlap
        assert own[i] & opp[i] == 0, "own and opp share a cell"
        # no position in the set is already finished
        assert not has_five(own[i]) and not has_five(opp[i])
    print(f"bitboards round-trip  ok ({n} records)")

    # --- last-move plane -------------------------------------------------
    li = d["last"]
    for i in range(0, n, 61):
        last_plane = (x[i, 2] > 0.5).reshape(-1)
        if li[i] >= 0:
            assert last_plane[li[i]], "last move not marked"
            assert last_plane.sum() == 1, "exactly one last move expected"
            # the last move was made by the *previous* player, i.e. the opponent
            # of the side to move, so it must land on the opp bitboard
            assert (opp[i] >> int(li[i])) & 1, "last move must be the opponent's stone"
        else:
            assert last_plane.sum() == 0
    print("last-move plane  ok")

    # --- dihedral --------------------------------------------------------
    assert len(DIHEDRAL) == 8
    for k, perm in enumerate(DIHEDRAL):
        assert sorted(perm.tolist()) == list(range(CELLS)), "perm is not a bijection"
        assert perm.tolist() == list(DIHEDRAL8[k]), f"perm {k} disagrees with spec module"
    x0 = augment(x, 0)
    assert np.array_equal(x0, x), "k=0 must be identity"
    for k in range(8):
        xk = augment(x, k)
        assert xk.shape == x.shape
        assert np.allclose(xk.sum(axis=(1, 2, 3)), x.sum(axis=(1, 2, 3)))
    print("dihedral augmentation  ok")

    # --- class weights ---------------------------------------------------
    w, counts = class_weights(d["cls"])
    exp = (w * counts).sum() / counts.sum()
    assert abs(exp - 1.0) < 1e-6, f"expected weight {exp}, should be 1.0"
    assert w[0] > w[2], "the rare class must get the larger weight"
    print(f"class weights  ok  (loss {w[0]:.2f}, win {w[2]:.2f})")

    print("ALL DATASET TESTS PASSED")


if __name__ == "__main__":
    main()
