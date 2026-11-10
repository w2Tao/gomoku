"""Train a value + policy network.

    loss = value_loss + lam * policy_loss

  value_loss  : the same weighted cross entropy used for baseline0, on one-hot
                solver labels and soft Monte-Carlo labels.
  policy_loss : cross entropy on the distilled 1-ply move, masked to legal
                cells so the network is never punished for the 30-odd occupied
                ones.

The trunk is warm-started from baseline0; only the policy head is new.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import paths
from dataset import DIHEDRAL, load, to_planes
from engine import CELLS, H, W
from model import N_CLASSES, ValuePolicyNet

PERMS = torch.tensor(np.stack(DIHEDRAL), dtype=torch.long)      # perm[dest] = src
INV = []
for _p in PERMS:
    inv = torch.empty_like(_p)
    inv[_p] = torch.arange(CELLS, dtype=_p.dtype)
    INV.append(inv)
INV = torch.stack(INV)                                          # inv[src] = dest


def load_policy_data(path):
    d = load(path)
    x = to_planes(d)                    # (N, 3, H, W) float32, shared layout
    return dict(
        x=torch.from_numpy(x),
        vt=torch.from_numpy(d["value_target"]),
        vw=torch.from_numpy(d["value_weight"]),
        mv=torch.from_numpy(d["move"].astype(np.int64)),
        val=torch.from_numpy(d["is_val"].astype(bool)),
        meta=d["meta"],
    )


def augment(x, mv, dev):
    b = x.shape[0]
    k = torch.randint(0, 8, (b,), device=dev)
    perm = PERMS.to(dev)[k]
    flat = x.reshape(b, 3, CELLS)
    out = torch.gather(flat, 2, perm.unsqueeze(1).expand(-1, 3, -1))
    nm = INV.to(dev)[k].gather(1, mv.unsqueeze(1)).squeeze(1)
    return out.reshape(b, 3, H, W), nm


def legal_mask(x):
    occ = x[:, 0].flatten(1) + x[:, 1].flatten(1)
    return occ < 0.5


@torch.no_grad()
def evaluate(net, X, VT, VW, MV, bs=1024):
    net.eval()
    vls, pls, correct, total = [], [], 0, 0
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs]
        v, p = net(xb)
        lv = -(VT[i:i + bs] * F.log_softmax(v, 1)).sum(1)
        vls.append((lv * VW[i:i + bs]).sum().item())
        m = legal_mask(xb)
        pm = p.masked_fill(~m, -1e9)
        pls.append(F.cross_entropy(pm, MV[i:i + bs]).item() * xb.shape[0])
        correct += (pm.argmax(1) == MV[i:i + bs]).sum().item()
        total += xb.shape[0]
    net.train()
    n = X.shape[0]
    return dict(value_loss=sum(vls) / n, policy_loss=sum(pls) / n,
                policy_acc=correct / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.data("policy1.npz"))
    ap.add_argument("--warm", default=paths.baseline_dir("baseline0") + "/model.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="policy loss weight")
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--head-channels", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--tag", default="pi1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = load_policy_data(args.data)
    tr = ~d["val"]
    X, VT, VW, MV = (d["x"][tr].to(dev), d["vt"][tr].to(dev),
                     d["vw"][tr].to(dev), d["mv"][tr].to(dev))
    Xv, VTv, VWv, MVv = (d["x"][~tr].to(dev), d["vt"][~tr].to(dev),
                         d["vw"][~tr].to(dev), d["mv"][~tr].to(dev))

    net = ValuePolicyNet(args.channels, args.blocks, args.head_channels).to(dev)
    warm = ""
    if args.warm and os.path.isfile(args.warm):
        ck = torch.load(args.warm, map_location=dev, weights_only=False)
        missing = net.load_state_dict(ck["state_dict"], strict=False)
        warm = f"  (warm-started from {os.path.basename(os.path.dirname(args.warm))}, " \
               f"new: {len(missing.missing_keys)} tensors)"
    n_par = sum(p.numel() for p in net.parameters())
    print(f"device      : {dev}")
    print(f"train / val : {int(tr.sum()):,} / {int((~tr).sum()):,}")
    print(f"model       : {args.channels}ch x {args.blocks} blocks, "
          f"{n_par:,} params{warm}")
    print(f"loss        : value + {args.lam} x policy")
    print()

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    out_dir = paths.run_dir(args.tag)
    n_tr = X.shape[0]
    hist, best = [], float("inf")
    t0 = time.time()

    for step in range(args.steps):
        idx = torch.randint(0, n_tr, (args.batch,), device=dev)
        xb, mb = augment(X[idx], MV[idx], dev)
        v, p = net(xb)
        lv = -(VT[idx] * F.log_softmax(v, 1)).sum(1)
        lv = (lv * VW[idx]).mean()
        pm = p.masked_fill(~legal_mask(xb), -1e9)
        lp = F.cross_entropy(pm, mb)
        loss = lv + args.lam * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            m = evaluate(net, Xv, VTv, VWv, MVv)
            m.update(step=step + 1, train_loss=loss.item(),
                     train_value=lv.item(), train_policy=lp.item(),
                     secs=round(time.time() - t0, 1))
            hist.append(m)
            star = ""
            if m["value_loss"] + args.lam * m["policy_loss"] < best:
                best = m["value_loss"] + args.lam * m["policy_loss"]
                torch.save({"state_dict": net.state_dict(),
                            "config": net.config, "step": step + 1,
                            "policy": True, "metrics": m},
                           os.path.join(out_dir, "best.pt"))
                star = "  *best"
            print(f"step {step+1:5d}  train {loss.item():.3f} "
                  f"(v {lv.item():.3f} p {lp.item():.3f})  "
                  f"val v {m['value_loss']:.3f} p {m['policy_loss']:.3f} "
                  f"acc {m['policy_acc']:.1%}  {m['secs']:.0f}s{star}",
                  flush=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist, "params": n_par}, f,
                  indent=2)
    print()
    print(f"checkpoint : {out_dir}/best.pt")
    print(f"total time : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
