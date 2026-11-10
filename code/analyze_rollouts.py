"""Compare heuristic rollouts against random rollouts and against the solver.

Three questions, in order of importance:

  1. Does the heuristic produce a value that actually varies with the position?
     Random rollouts concentrate near 0.5, which is why the label said little.
  2. On positions where the solver knows the answer, does the rollout value agree
     with it? That is the only ground-truth check available for a rollout policy.
  3. Is the new label distribution usable -- spread across the range, or piled up
     at the extremes like the solver labels?
"""
import sys

import numpy as np

import paths
from dataset import bitboards, load


def v_of(counts):
    p = counts.astype(np.float32) / counts.sum(1, keepdims=True)
    return p[:, 2] - p[:, 0]


def hist(v, edges=(-1.01, -0.75, -0.25, 0.25, 0.75, 1.01)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(((v >= lo) & (v < hi)).mean())
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mc3.npz"
    mc1 = load(paths.data("mc1.npz"))
    mc3 = load(paths.data(name))
    batch = load(paths.data("batch1.npz"))

    v1 = v_of(mc1["counts"])
    v3 = v_of(mc3["counts"])
    print(f"mc1 : {len(v1):,} positions, uniform-random rollouts")
    print(f"mc3 : {len(v3):,} positions, {mc3['meta']['rollout_policy']}")
    print()

    print("=" * 74)
    print("1. value distribution")
    print("=" * 74)
    print(f"{'':<30}{'mean':>8}{'std':>8}{'|V|>0.5':>9}{'|V|<0.1':>9}")
    for label, v in (("random rollouts (mc1)", v1),
                     ("heuristic rollouts (mc3)", v3),
                     ("solver labels (batch1)", None)):
        if v is None:
            continue
        print(f"{label:<30}{v.mean():>+8.3f}{v.std():>8.3f}"
              f"{100*(np.abs(v)>0.5).mean():>8.0f}%"
              f"{100*(np.abs(v)<0.1).mean():>8.0f}%")
    sv = batch["cls"].astype(np.float32) - 1.0     # 0->-1, 2->+1
    print(f"{'solver labels (batch1)':<30}{sv.mean():>+8.3f}{sv.std():>8.3f}"
          f"{100*(np.abs(sv)>0.5).mean():>8.0f}%"
          f"{100*(np.abs(sv)<0.1).mean():>8.0f}%")
    print()
    print("histogram  (-1,-.75] (-.75,-.25] (-.25,.25] (.25,.75] (.75,1]")
    for label, v in (("random", v1), ("heuristic", v3)):
        print(f"  {label:<10}" + "".join(f"{x:>11.0%}" for x in hist(v)))

    print()
    print("=" * 74)
    print("2. the same positions under both rollout policies")
    print("=" * 74)
    m = mc3["src_name"] == 0
    if m.any():
        a = v3[m]
        b = v1[mc3["src_index"][m]]
        print(f"positions      : {m.sum()}")
        print(f"random         : mean {b.mean():+.3f}  std {b.std():.3f}")
        print(f"heuristic      : mean {a.mean():+.3f}  std {a.std():.3f}")
        print(f"correlation    : {np.corrcoef(a, b)[0,1]:+.2f}")
        print(f"mean |diff|    : {np.abs(a - b).mean():.3f}")
        print(f"sign flipped   : {100*((a > 0) != (b > 0)).mean():.0f}%")

    print()
    print("=" * 74)
    print("3. solver as ground truth")
    print("=" * 74)
    s = mc3["src_name"] == 1
    if s.any():
        cls = batch["cls"][mc3["src_index"][s]]
        va = v3[s]
        for c, label in ((2, "solver says WIN"), (0, "solver says LOSS")):
            mm = cls == c
            if mm.any():
                print(f"{label:<20} n={mm.sum():>3}   "
                      f"rollout V mean {va[mm].mean():+.3f}   "
                      f"frac V>0 {100*(va[mm] > 0).mean():5.0f}%")
        agree = ((va > 0) & (cls == 2)) | ((va < 0) & (cls == 0))
        print(f"sign agreement      {agree.mean():.0%}")

    own, opp = bitboards(mc3)
    ply = np.array([bin(o | p).count("1") for o, p in zip(own, opp)])
    print()
    print("=" * 74)
    print("4. by ply (heuristic rollouts)")
    print("=" * 74)
    print(f"{'ply':>8}{'n':>7}{'mean V':>10}{'std V':>9}{'|V|>0.5':>9}")
    for lo in range(0, 90, 15):
        mm = (ply >= lo) & (ply < lo + 15)
        if mm.sum() >= 5:
            v = v3[mm]
            print(f"{lo:>4}-{lo+14:<3}{mm.sum():>7}{v.mean():>+10.3f}"
                  f"{v.std():>9.3f}{100*(np.abs(v)>0.5).mean():>8.0f}%")


if __name__ == "__main__":
    main()
