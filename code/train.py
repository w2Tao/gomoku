"""Train the value network on a mix of exact and Monte-Carlo labels.

Two label sources, one soft-target objective:

    loss = mean_i  w_i * CE( softmax(V(s_i)), t_i )

  solver set (`--data`) : t_i is one-hot (V* = +-1), w_i is the class weight.
                          Exact, but only exists at the extremes.
  MC set     (`--mc`)   : t_i is the rollout outcome distribution, w_i is a flat
                          `--mc-weight`. Weak, but graded -- which is the whole
                          point: it supplies the middle of the value range that
                          the solver structurally cannot label.

Validation reports both halves separately, and the metric that actually matters
is not accuracy but whether a trained net beats a randomly-initialised one at
the same search depth (`eval_play.py`).

    python train.py --steps 6000 --mc data/mc1.npz --mc-weight 0.5 --tag v2
"""
from __future__ import annotations

import paths
from engine import CELLS, H, W

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset import DIHEDRAL, class_weights, load, to_planes
from model import N_CLASSES, ValueNet

PERMS = torch.tensor(np.stack(DIHEDRAL), dtype=torch.long)


def augment_batch(x, dev):
    b = x.shape[0]
    k = torch.randint(0, 8, (b,), device=dev)
    perm = PERMS.to(dev)[k]
    flat = x.reshape(b, 3, CELLS)
    out = torch.gather(flat, 2, perm.unsqueeze(1).expand(-1, 3, -1))
    return out.reshape(b, 3, H, W)


def build_mix(solver_path, mc_path, mc_weight, shuffle_mc=False, seed=0):
    xs, ts, ws, vs, ss = [], [], [], [], []
    info = {}
    if solver_path:
        d = load(solver_path)
        cls = d["cls"].astype(np.int64)
        t = np.zeros((len(cls), N_CLASSES), np.float32)
        t[np.arange(len(cls)), cls] = 1.0
        cw, cnt = class_weights(cls)
        xs.append(to_planes(d))
        ts.append(t)
        ws.append(cw[cls].astype(np.float32))
        vs.append(d["is_val"].astype(bool))
        ss.append(np.zeros(len(cls), np.int8))
        info["solver"] = dict(n=int(len(cls)), counts=cnt.tolist(),
                              class_weight=cw.tolist(), path=solver_path)
    if mc_path:
        d = load(mc_path)
        c = d["counts"].astype(np.float32)     # columns are [loss, draw, win]
        t = c / c.sum(1, keepdims=True)
        if shuffle_mc:
            # placebo: identical data volume and identical target distribution,
            # but the label is attached to the wrong board. If the gain survives
            # this, it came from the output prior, not from positional information.
            rs = np.random.RandomState(seed)
            t = t[rs.permutation(len(t))]
        xs.append(to_planes(d))
        ts.append(t)
        ws.append(np.full(len(t), mc_weight, np.float32))
        vs.append(d["is_val"].astype(bool))
        ss.append(np.ones(len(t), np.int8))
        info["mc"] = dict(n=int(len(t)), rollouts=d["meta"].get("rollouts"),
                          path=mc_path, weight=mc_weight)
    return (np.concatenate(xs), np.concatenate(ts), np.concatenate(ws),
            np.concatenate(vs), np.concatenate(ss), info)


@torch.no_grad()
def evaluate(model, X, T, W, S, bs=1024):
    model.eval()
    out = []
    for i in range(0, X.shape[0], bs):
        out.append(model(X[i:i + bs]))
    logits = torch.cat(out)
    logp = F.log_softmax(logits, 1)
    per = -(T * logp).sum(1)
    res = {"loss": (per * W).mean().item()}

    sm = S == 0
    if sm.any():
        y = T[sm].argmax(1)
        pred = logits[sm].argmax(1)
        rec = [((pred == c) & (y == c)).sum().item() / max((y == c).sum().item(), 1)
               for c in range(N_CLASSES) if (y == c).any()]
        res["solver_acc"] = (pred == y).float().mean().item()
        res["solver_bal"] = float(np.mean(rec)) if rec else 0.0

    mm = S == 1
    if mm.any():
        p = torch.softmax(logits[mm], 1)
        v_pred = (p[:, 2] - p[:, 0]).cpu().numpy()
        v_true = (T[mm][:, 2] - T[mm][:, 0]).cpu().numpy()
        res["mc_mae"] = float(np.abs(v_pred - v_true).mean())
        res["mc_corr"] = float(np.corrcoef(v_pred, v_true)[0, 1])
        res["mc_std_pred"] = float(v_pred.std())
        res["mc_std_true"] = float(v_true.std())
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.data("batch1.npz"))
    ap.add_argument("--mc", default=None)
    ap.add_argument("--mc-weight", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--head-channels", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--mc-shuffle", action="store_true",
                    help="placebo: permute MC targets across positions")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xn, Tn, Wn, Vn, Sn, info = build_mix(args.data, args.mc, args.mc_weight,
                                         args.mc_shuffle, args.seed)
    X = torch.from_numpy(Xn).to(dev)
    T = torch.from_numpy(Tn).to(dev)
    Wt = torch.from_numpy(Wn).to(dev)
    tr = ~Vn
    va = Vn
    Xtr, Ttr, Wtr = X[tr], T[tr], Wt[tr]
    Xva, Tva, Wva, Sva = X[va], T[va], Wt[va], torch.from_numpy(Sn[va]).to(dev)

    model = ValueNet(args.channels, args.blocks, args.head_channels).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device      : {dev}")
    print(f"train / val : {int(tr.sum()):,} / {int(va.sum()):,}")
    for k, v in info.items():
        print(f"  {k:7s} : {v['n']:,} records  ({v['path']})")
    print(f"model       : {args.channels}ch x {args.blocks} blocks, "
          f"{n_par:,} params")
    print(f"augmentation: {'off' if args.no_aug else '8-fold per sample'}")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    paths.run_dir(args.tag)

    hist = []
    best = float("inf")
    t0 = time.time()
    n_tr = Xtr.shape[0]

    for step in range(args.steps):
        idx = torch.randint(0, n_tr, (args.batch,), device=dev)
        xb = Xtr[idx]
        if not args.no_aug:
            xb = augment_batch(xb, dev)
        logp = F.log_softmax(model(xb), 1)
        per = -(Ttr[idx] * logp).sum(1)
        loss = (per * Wtr[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            m = evaluate(model, Xva, Tva, Wva, Sva)
            m["step"] = step + 1
            m["train_loss"] = loss.item()
            m["secs"] = round(time.time() - t0, 1)
            hist.append(m)
            star = ""
            if m["loss"] < best:
                best = m["loss"]
                torch.save({"state_dict": model.state_dict(),
                            "config": model.config, "step": step + 1,
                            "metrics": m}, os.path.join(paths.run_dir(args.tag), "best.pt"))
                star = "  *best"
            extra = ""
            if "solver_bal" in m:
                extra += f" bal {m['solver_bal']:.3f}"
            if "mc_mae" in m:
                extra += f" mcMAE {m['mc_mae']:.3f} r {m['mc_corr']:.2f}"
            print(f"step {step+1:5d}  train {loss.item():.4f}  "
                  f"val {m['loss']:.4f}{extra}  {m['secs']:.0f}s{star}",
                  flush=True)

    with open(os.path.join(paths.run_dir(args.tag), "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "info": info, "history": hist,
                   "params": n_par, "best_val_loss": best}, f, indent=2)
    print()
    print(f"checkpoint : runs/{args.tag}/best.pt")
    print(f"total time : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
