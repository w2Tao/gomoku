"""Is the heuristic-rollout label usable in the opening?

600 positions is far too few to train a value function, so this is a
learnability test rather than a strength test. One question:

    On held-out positions -- especially early ones -- does a model trained on the
    heuristic-rollout label predict that label better than baseline0 does?

Baselines that matter:
    constant    predicts the training mean; a model that cannot beat it has
                learned nothing position-dependent.
    baseline0   trained on solver + random-rollout labels, so the gap measures
                what the new label says that the old one did not.

The split is by game: consecutive plies of one game are nearly the same board.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

import paths
from dataset import bitboards, load, to_planes
from model import ValueNet

TAG = sys.argv[1] if len(sys.argv) > 1 else "mc3.npz"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
VAL_FRAC = 5


def v_of(counts):
    p = counts.astype(np.float32) / counts.sum(1, keepdims=True)
    return p[:, 2] - p[:, 0], p


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load(paths.data(TAG))
    mc1 = load(paths.data("mc1.npz"))
    b1 = load(paths.data("batch1.npz"))

    game = np.empty(len(d["counts"]), dtype=np.int64)
    for row, (src, idx) in enumerate(zip(d["src_name"], d["src_index"])):
        game[row] = (mc1 if src == 0 else b1)["game_id"][idx]
    own, opp = bitboards(d)
    ply = np.array([bin(o | p).count("1") for o, p in zip(own, opp)])

    x = torch.from_numpy(to_planes(d)).to(dev)
    v_t, p_t = v_of(d["counts"])
    t = torch.from_numpy(p_t).to(dev)
    vt = torch.from_numpy(v_t).to(dev)

    val_np = (game % VAL_FRAC == 0)
    val = torch.from_numpy(val_np).to(dev)
    tr = ~val
    early = torch.from_numpy(ply <= 14).to(dev)
    late = ~early
    idx_tr = torch.nonzero(tr).squeeze(1)

    print(f"positions : {len(v_t)}   train {int(tr.sum())} / val {int(val.sum())} "
          f"({len(set(game[tr.cpu().numpy()]))} / "
          f"{len(set(game[val_np]))} games)")
    print(f"opening   : train {int((tr & early).sum())} / "
          f"val {int((val & early).sum())}  (ply <= 14)")
    print()

    def mae(v_pred, mask):
        return float((v_pred[mask] - vt[mask]).abs().mean())

    rows = []
    const = torch.full_like(vt, float(v_t[~val_np].mean()))
    rows.append(("constant (train mean)", const))

    ck = torch.load(paths.baseline_dir("baseline0") + "/model.pt",
                    map_location=dev, weights_only=False)

    base = ValueNet(**ck["config"]).to(dev)
    base.load_state_dict(ck["state_dict"])
    base.eval()
    with torch.no_grad():
        p = torch.softmax(base(x), 1)
        rows.append(("baseline0 (unchanged)", p[:, 2] - p[:, 0]))

    def tune(reset: bool, steps: int, lr: float, tag: str):
        net = ValueNet(**ck["config"]).to(dev)
        net.load_state_dict(ck["state_dict"])
        if reset:
            torch.manual_seed(1)
            torch.nn.init.kaiming_uniform_(net.head[4].weight, a=5 ** 0.5)
            torch.nn.init.zeros_(net.head[4].bias)
        for p_ in net.stem.parameters():
            p_.requires_grad = False
        for p_ in net.body.parameters():
            p_.requires_grad = False
        net.eval()                        # freezes BatchNorm statistics too
        opt = torch.optim.AdamW(net.head.parameters(), lr=lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        for step in range(steps):
            sel = idx_tr[torch.randint(0, len(idx_tr), (64,), generator=g)]
            loss = -(t[sel] * F.log_softmax(net(x[sel]), 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % 200 == 0:
                print(f"  {tag} step {step+1:4d}  loss {loss.item():.4f}")
        with torch.no_grad():
            p = torch.softmax(net(x), 1)
        return p[:, 2] - p[:, 0]

    rows.append(("baseline0 head, fine-tuned", tune(False, STEPS, 1e-3, "finetune")))
    rows.append(("fresh head, frozen trunk", tune(True, STEPS, 3e-3, "fresh")))

    print()
    print(f"{'model':<28}{'val MAE':>10}{'ply<=14':>10}{'ply>14':>10}")
    print("-" * 58)
    for name, v in rows:
        print(f"{name:<28}{mae(v, val):>10.3f}{mae(v, val & early):>10.3f}"
              f"{mae(v, val & late):>10.3f}")
    print()
    print(f"target std: overall {v_t[val_np].std():.3f}   "
          f"ply<=14 {v_t[val_np][ply[val_np] <= 14].std():.3f}")


if __name__ == "__main__":
    main()
