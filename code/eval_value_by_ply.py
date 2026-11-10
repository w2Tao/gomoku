"""Compare value checkpoints on the blind-region labels, by game phase.

The target here is V^pi under the heuristic, which is the only label that exists
in the solver-blind region. Read the MAE numbers with that in mind: a model
trained harder toward this target will win this particular comparison by
construction, so the interesting part is *where* the gap sits (early plies, where
baseline0 had no supervision at all) rather than the single aggregate number.
"""
import sys

import numpy as np
import torch

import paths
from dataset import load, to_planes
from play import load_net

MC = sys.argv[1] if len(sys.argv) > 1 else "mc4.npz"
TAGS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["baseline0", "v3"]
BANDS = [(0, 14), (15, 29), (30, 44), (45, 59), (60, 74), (75, 224)]


def v_of(counts):
    p = counts.astype(np.float32) / counts.sum(1, keepdims=True)
    return p[:, 2] - p[:, 0]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load(paths.data(MC))
    v_t = v_of(d["counts"])
    ply = d["ply"].astype(int) if "ply" in d else np.zeros(len(v_t), int)
    val = d["is_val"] == 1
    x = torch.from_numpy(to_planes(d)[val]).to(dev)
    y = torch.from_numpy(v_t[val]).to(dev)
    p_val = ply[val]
    print(f"{MC}: {val.sum()} held-out positions, "
          f"target std {v_t[val].std():.3f}")
    print()

    hdr = f"{'checkpoint':<16}{'MAE':>8}" + "".join(
        f"{f'ply {a}-{b}':>12}" for a, b in BANDS)
    print(hdr)
    print("-" * len(hdr))
    for tag in TAGS:
        path = (paths.baseline_dir(tag) + "/model.pt" if tag == "baseline0"
                else f"{paths.run_dir(tag)}/best.pt")
        net, ck = load_net(path, dev)
        with torch.no_grad():
            p = torch.softmax(net(x), 1)
            v = (p[:, 2] - p[:, 0]).cpu().numpy()
        err = np.abs(v - v_t[val])
        cells = [f"{err.mean():>8.3f}"]
        for a, b in BANDS:
            m = (p_val >= a) & (p_val <= b)
            cells.append(f"{err[m].mean():>12.3f}" if m.sum() >= 5 else
                         f"{'-':>12}")
        print(f"{tag:<16}" + "".join(cells))
    print()
    print("n per band        " + "".join(
        f"{int(((p_val >= a) & (p_val <= b)).sum()):>12}" for a, b in BANDS))


if __name__ == "__main__":
    main()
