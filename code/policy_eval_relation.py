"""Does RAPFI's played move have the best child eval at a fixed 1 s budget?

For every sampled position we ask RAPFI for its move at 1 s, then ask it to
evaluate every legal child at 1 s. The child is the opponent to move, so
"RAPFI's move is best" means its child eval is the **minimum** among candidates,
not the maximum.

The comparison is within a fixed budget; it is not a claim about the engine's
internal root values at another depth.

    python code/policy_eval_relation.py --positions 12 --workers 8 --time-ms 1000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, "code")

from engine import CELLS, W, has_five                       # noqa: E402
from fetch_engines import ENGINE                            # noqa: E402
from pbrain import Pbrain, find_exe, rapfi_targets          # noqa: E402


def cells(bb):
    out = []
    while bb:
        low = bb & -bb
        out.append(low.bit_length() - 1)
        bb ^= low
    return out


def board_lines(black, white):
    """Emit stones in a legal alternating order.

    `BOARD` interprets who=1/2 relative to the first stone and inserts PASS moves
    when consecutive lines do not alternate. Grouping all black stones before all
    white stones therefore inserts passes and changes the history (and the
    `last` move) the engine sees. Alternating colours avoids that entirely.
    """
    blacks, whites = cells(black), cells(white)
    lines = []
    for i in range(max(len(blacks), len(whites))):
        if i < len(blacks):
            c = blacks[i]
            lines.append(f"{c % W},{c // W},1")
        if i < len(whites):
            c = whites[i]
            lines.append(f"{c % W},{c // W},2")
    return lines


def query(eng, black, white):
    """Set BOARD and return (move_cell, eval). Eval is side-to-move."""
    eng._send("BOARD")
    for line in board_lines(black, white):
        eng._send(line)
    eng._send("DONE")
    mv = eng._read_move(5000)
    ev = rapfi_targets(eng.last_messages)[0]
    x, y = mv
    return y * W + x, ev


def one_position(spec, exe, time_ms):
    idx, black, white, own, opp, ply = spec
    eng = Pbrain(exe, time_ms=time_ms, stall_s=10.0)
    try:
        eng.start()
        chosen, root_eval = query(eng, black, white)
        results = []
        for c in range(CELLS):
            if (black | white) >> c & 1:
                continue
            if has_five(own | (1 << c)):
                # our move already wins; it is the best possible child
                results.append((c, -10 ** 9))
                continue
            if own.bit_count() == opp.bit_count():
                b2, w2 = black | (1 << c), white
            else:
                b2, w2 = black, white | (1 << c)
            try:
                _, child_eval = query(eng, b2, w2)
            except Exception as exc:                      # noqa: BLE001
                print(f"  pos {idx} candidate {c}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            if child_eval is not None:
                results.append((c, child_eval))
        vals = [v for _, v in results]
        if not vals:
            return None
        chosen_ev = next((v for c, v in results if c == chosen), None)
        if chosen_ev is None:
            return None
        rank = 1 + sum(v < chosen_ev for v in vals)
        return dict(idx=idx, ply=ply, chosen=chosen, root_eval=root_eval,
                    chosen_eval=chosen_ev, best_eval=min(vals), n=len(vals),
                    rank=rank)
    finally:
        eng.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-ms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="policy_eval_relation.json")
    args = ap.parse_args()

    exe = find_exe(os.path.join("data", "engines", "RAPFI25"),
                   ENGINE()["RAPFI25"]["prefer"])
    d = np.load("data/gomocup15.npz", allow_pickle=False)
    rng = np.random.default_rng(args.seed)
    cand = np.nonzero((d["ply"] >= 4) & (d["ply"] <= 16))[0]
    cand = rng.choice(cand, size=min(len(cand), args.positions * 8),
                      replace=False)
    specs = []
    for i in cand:
        own = int(sum(int(v) << (64 * k)
                      for k, v in enumerate(d["own"][i])))
        opp = int(sum(int(v) << (64 * k)
                      for k, v in enumerate(d["opp"][i])))
        nc, no = own.bit_count(), opp.bit_count()
        if nc == no:
            black, white = own, opp
        elif nc == no + 1:
            black, white = opp, own
        else:
            continue
        if has_five(black) or has_five(white):
            continue
        specs.append((len(specs), black, white, own, opp, int(d["ply"][i])))
        if len(specs) >= args.positions:
            break

    print(f"testing {len(specs)} positions, all legal children, "
          f"{args.time_ms} ms, {args.workers} workers", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one_position, s, exe, args.time_ms) for s in specs]
        for f in as_completed(futs):
            r = f.result()
            if r:
                out.append(r)
                print(f"  pos {r['idx']:>2} ply {r['ply']:>2}  "
                      f"chosen ({r['chosen'] % W},{r['chosen'] // W})  "
                      f"child {r['chosen_eval']:>6}  best {r['best_eval']:>6}  "
                      f"rank {r['rank']:>3}/{r['n']:<3}  root {r['root_eval']}",
                      flush=True)

    if not out:
        raise SystemExit("no completed positions")
    ranks = [r["rank"] for r in out]
    n1 = sum(r == 1 for r in ranks)
    n3 = sum(r <= 3 for r in ranks)
    print()
    print(f"positions          : {len(out)}")
    print(f"chosen is min eval : {n1}/{len(out)} = {n1/len(out):.1%}")
    print(f"chosen in top 3    : {n3}/{len(out)} = {n3/len(out):.1%}")
    print(f"mean rank          : {statistics.mean(ranks):.2f}")
    print(f"median rank        : {statistics.median(ranks):.1f}")
    valid = [r for r in out if r["n"] >= 10]
    if valid:
        vr = [r["rank"] for r in valid]
        print(f"positions with >=10 children: {len(valid)}")
        print(f"  chosen is min    : {sum(r == 1 for r in vr)}/{len(valid)}")
        print(f"  chosen in top 3  : {sum(r <= 3 for r in vr)}/{len(valid)}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
