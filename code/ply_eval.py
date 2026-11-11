"""Is the distilled model worse in the opening?

Distillation data is collected from games where our side opens with random moves, so
the positions are skew: plenty of early ones, but reached by bad play. The risk is
that the policy learns "how to defend a mess" instead of "how to open".

The direct check is agreement with the move an engine actually chose, split by ply.
Early plies are the opening; a model that got worse there shows up as a drop in the
first bucket even while the totals improve.

    python code/ply_eval.py baseline,d2,d3
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "code")

import baselines                                       # noqa: E402
import paths                                           # noqa: E402
from dataset import load, to_planes                    # noqa: E402

BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 45), (46, 999)]


def main():
    models = (sys.argv[1] if len(sys.argv) > 1 else "baseline,d2,d3").split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = load(paths.data("gomocup15.npz"))
    sel = np.nonzero(d["is_val"])[0]
    x = torch.from_numpy(to_planes(d)[sel]).to(dev)
    mv = torch.from_numpy(d["move"][sel].astype(np.int64)).to(dev)
    ply = d["ply"][sel].astype(np.int64)
    occ = torch.from_numpy((d["own"][sel] | d["opp"][sel])).unsqueeze(1)

    print(f"{len(sel):,} held-out positions\n")
    head = f"{'model':<10}" + "".join(f"{f'{a}-{b}' if b < 999 else f'{a}+':>10}"
                                      for a, b in BUCKETS) + f"{'all':>9}"
    print(head)
    print("-" * len(head))

    for mid in models:
        net, _ = baselines.load_baseline(mid, dev)
        picked = np.empty(len(sel), dtype=np.int64)
        with torch.no_grad():
            # batched: 50k positions at once does not fit on the GPU
            for i in range(0, len(sel), 2048):
                xb = x[i:i + 2048]
                out = net(xb)
                logits = out[1] if isinstance(out, tuple) else out
                occ_mask = (xb[:, 0] + xb[:, 1]).reshape(xb.shape[0], -1) > 0.5
                picked[i:i + 2048] = (
                    logits.masked_fill(occ_mask, -1e9).argmax(1).cpu().numpy())
        mv_np = mv.cpu().numpy()
        rows = f"{mid:<10}"
        tot_ok = 0
        for a, b in BUCKETS:
            m = (ply >= a) & (ply <= b)
            if m.sum() == 0:
                rows += f"{'-':>10}"
                continue
            acc = float((picked[m] == mv_np[m]).mean())
            tot_ok += int((picked[m] == mv_np[m]).sum())
            rows += f"{acc:>9.1%} "
        rows += f"{tot_ok/len(sel):>8.1%}"
        print(rows)


if __name__ == "__main__":
    main()
