"""Train a pure scalar eval network on RAPFI `Eval` labels.

    target = tanh(Eval / 400)

The 400 follows from RAPFI's default `winrate = sigmoid(Eval / 200)`:
`2 * winrate - 1 = tanh(Eval / 400)`. Mate evals are clamped by the tanh target.

    python code/train_eval.py --data data/eval20k.npz --steps 12000 --tag eval20k
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import paths
from dataset import DIHEDRAL, load, to_planes
from engine import CELLS, H, W
from model import ValueEvalNet

PERMS = torch.tensor(np.stack(DIHEDRAL), dtype=torch.long)


def augment_batch(x, dev):
    b = x.shape[0]
    k = torch.randint(0, 8, (b,), device=dev)
    perm = PERMS.to(dev)[k]
    flat = x.reshape(b, 3, CELLS)
    out = torch.gather(flat, 2, perm.unsqueeze(1).expand(-1, 3, -1))
    return out.reshape(b, 3, H, W)


@torch.no_grad()
def evaluate(net, X, Y, bs=1024):
    net.eval()
    preds, ys = [], []
    for i in range(0, X.shape[0], bs):
        preds.append(net(X[i:i + bs]).cpu())
        ys.append(Y[i:i + bs].cpu())
    p = torch.cat(preds).numpy()
    y = torch.cat(ys).numpy()
    net.train()
    mae = float(np.abs(p - y).mean())
    corr = float(np.corrcoef(p, y)[0, 1]) if len(p) > 1 else 0.0
    return dict(mae=mae, corr=corr, n=len(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval20k.npz")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-temp", type=float, default=400.0)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--head-channels", type=int, default=48)
    ap.add_argument("--warm", default="", help="optional value/policy checkpoint")
    ap.add_argument("--tag", default="eval20k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = load(args.data if os.path.isfile(args.data) else paths.data(args.data))
    x = torch.from_numpy(to_planes(d))
    ev = d["eval"].astype(np.float32)
    y = np.tanh(np.clip(ev / args.eval_temp, -30, 30))
    val = d["is_val"].astype(bool)
    X, Y = x[~val].to(dev), torch.from_numpy(y[~val]).to(dev)
    Xv, Yv = x[val].to(dev), torch.from_numpy(y[val]).to(dev)

    net = ValueEvalNet(args.channels, args.blocks, args.head_channels).to(dev)
    warm = ""
    if args.warm and os.path.isfile(args.warm):
        ck = torch.load(args.warm, map_location=dev, weights_only=False)
        miss = net.load_state_dict(ck["state_dict"], strict=False)
        warm = (f"  (warm-started from {os.path.basename(os.path.dirname(args.warm))}, "
                f"missing: {len(miss.missing_keys)})")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    out_dir = paths.run_dir(args.tag)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"device      : {dev}")
    print(f"train / val : {int((~val).sum()):,} / {int(val.sum()):,}")
    print(f"model       : {args.channels}ch x {args.blocks} blocks, "
          f"{n_par:,} params{warm}")
    print(f"target      : tanh(Eval / {args.eval_temp:g}), MSE")
    print()

    best = float("inf")
    hist = []
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, X.shape[0], (args.batch,), device=dev)
        xb = augment_batch(X[idx], dev)
        pred = net(xb)
        loss = F.mse_loss(pred, Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            m = evaluate(net, Xv, Yv)
            m.update(step=step + 1, train_loss=loss.item(),
                     secs=round(time.time() - t0, 1))
            hist.append(m)
            star = ""
            if m["mae"] < best:
                best = m["mae"]
                torch.save({"state_dict": net.state_dict(),
                            "config": net.config, "step": step + 1,
                            "metrics": m, "eval_temp": args.eval_temp},
                           os.path.join(out_dir, "best.pt"))
                star = "  *best"
            print(f"step {step+1:5d}  train {loss.item():.4f}  "
                  f"val mae {m['mae']:.4f}  corr {m['corr']:+.4f}  "
                  f"{m['secs']:.0f}s{star}", flush=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "history": hist, "params": n_par}, f,
                  indent=2)
    print()
    print(f"checkpoint : {out_dir}/best.pt")
    print(f"total time : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
