"""Collect RAPFI `Eval` labels on arbitrary positions from the baseline corpus.

This is the data source for a pure eval/value network: each sampled position is
set with `BOARD` (stones must alternate colours, or the engine inserts implicit
PASS moves), searched at a fixed time, and labelled with the side-to-move
`Eval`. No move policy is recorded.

    python code/build_eval_dataset.py --positions 20000 --workers 12 --time-ms 1000
    python code/build_eval_dataset.py --verify 20 --time-ms 1000

The verify mode asks the same position twice and reports exact-eval / move
agreement; it does not write a dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "code")

import paths                                              # noqa: E402
from dataset import pack_boards                           # noqa: E402
from engine import CELLS, W, has_five                     # noqa: E402
from fetch_engines import ENGINE                          # noqa: E402
from pbrain import Pbrain, find_exe, rapfi_targets        # noqa: E402

_DEPTH_RE = re.compile(r"Depth\s+(\d+)-(\d+)")


def cells(bb):
    out = []
    while bb:
        low = bb & -bb
        out.append(low.bit_length() - 1)
        bb ^= low
    return out


def board_lines(black, white):
    """Alternate black/white stones so BOARD does not insert PASS moves."""
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
    """BOARD, DONE, return (eval, depth, move_cell). Eval is side-to-move."""
    eng._send("BOARD")
    for line in board_lines(black, white):
        eng._send(line)
    eng._send("DONE")
    mv = eng._read_move(8000)
    ev = rapfi_targets(eng.last_messages)[0]
    depths = _DEPTH_RE.findall("\n".join(eng.last_messages))
    depth = int(depths[-1][0]) if depths else 0
    return ev, depth, mv[1] * W + mv[0]


def load_specs(n, seed):
    d = np.load(paths.data("gomocup15.npz"), allow_pickle=False)
    ply = d["ply"]
    bins = ((2, 10), (11, 20), (21, 30), (31, 45), (46, 999))
    rng = np.random.default_rng(seed)
    per = max(1, n // len(bins))
    chosen = []
    for lo, hi in bins:
        pool = np.nonzero((ply >= lo) & (ply <= hi))[0]
        if not len(pool):
            continue
        k = min(per, len(pool))
        chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    if len(chosen) < n:
        pool = np.arange(len(ply))
        extra = rng.choice(pool, size=min(n - len(chosen), len(pool)),
                           replace=False).tolist()
        chosen.extend(extra)
    chosen = list(dict.fromkeys(chosen))[:n]
    rng.shuffle(chosen)

    specs = []
    for i in chosen:
        own = int(sum(int(v) << (64 * k) for k, v in enumerate(d["own"][i])))
        opp = int(sum(int(v) << (64 * k) for k, v in enumerate(d["opp"][i])))
        nc, no = own.bit_count(), opp.bit_count()
        if nc == no:
            black, white = own, opp
        elif no == nc + 1:
            # side to move is white: black (opponent) has one more stone
            black, white = opp, own
        else:
            continue
        if has_five(black) or has_five(white):
            continue
        specs.append(dict(idx=i, black=black, white=white, own=own, opp=opp,
                          last=int(d["last"][i]), ply=int(d["ply"][i]),
                          gid=int(d["game_id"][i])))
    return specs


def verify(specs, n, exe, time_ms):
    rng = random.Random(0)
    sample = rng.sample(specs, min(n, len(specs)))
    same_eval = same_move = 0
    diffs, depths = [], []
    for s in sample:
        vals = []
        for _ in range(2):
            # Fresh process per query: reusing one engine (even with RESTART)
            # still changes the time-limited search, especially near mates.
            eng = Pbrain(exe, time_ms=time_ms, stall_s=15.0)
            try:
                eng.start()
                vals.append(query(eng, s["black"], s["white"]))
            finally:
                eng.close()
        e1, d1, m1 = vals[0]
        e2, d2, m2 = vals[1]
        if e1 is not None and e2 is not None:
            same_eval += int(e1 == e2)
            diffs.append(abs(e1 - e2))
            depths.append((d1, d2))
        same_move += int(m1 == m2)
    diffs = np.asarray(diffs, dtype=np.int64)
    print(f"verified {len(sample)} positions, two 1s searches each, fresh engines")
    print(f"  exact eval match : {same_eval}/{len(sample)} = {same_eval/len(sample):.1%}")
    print(f"  move match       : {same_move}/{len(sample)} = {same_move/len(sample):.1%}")
    print(f"  eval |diff| mean : {diffs.mean():.2f}  max {diffs.max() if len(diffs) else 0}")
    if depths:
        print(f"  depth pairs      : {depths[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--time-ms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval20k.npz")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--verify", type=int, default=0,
                    help="verify N positions twice, then exit")
    args = ap.parse_args()

    exe = find_exe(os.path.join("data", "engines", "RAPFI25"),
                   ENGINE()["RAPFI25"]["prefer"])
    specs = load_specs(args.positions if not args.verify else max(args.verify, 20),
                       args.seed)
    if args.shards > 1:
        specs = specs[args.shard::args.shards]
        args.workers = 1
    print(f"sampled {len(specs)} positions from gomocup15", flush=True)
    if args.verify:
        verify(specs, args.verify, exe, args.time_ms)
        return 0

    chunks = np.array_split(np.arange(len(specs)), args.workers)
    lock = threading.Lock()
    done = 0
    errors = 0
    rows = []

    def work(chunk):
        nonlocal done, errors
        local = []
        err = 0
        for pos in chunk:
            s = specs[int(pos)]
            eng = Pbrain(exe, time_ms=args.time_ms, stall_s=15.0)
            try:
                eng.start()
                ev, depth, _mv = query(eng, s["black"], s["white"])
                if ev is None:
                    err += 1
                else:
                    local.append((s["own"], s["opp"], s["last"], s["ply"],
                                  int(ev), depth, s["gid"]))
            except Exception as exc:                      # noqa: BLE001
                err += 1
                print(f"  pos {s['idx']}: {type(exc).__name__}: {exc}",
                      flush=True)
            finally:
                try:
                    eng.close()
                except Exception:
                    pass
            with lock:
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(specs)}  errors {errors + err}",
                          flush=True)
        with lock:
            errors += err
        return local

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(work, chunks):
            rows.extend(part)

    if not rows:
        raise SystemExit("no eval labels")
    own = [r[0] for r in rows]
    opp = [r[1] for r in rows]
    last = np.asarray([r[2] for r in rows], dtype=np.int16)
    ply = np.asarray([r[3] for r in rows], dtype=np.int16)
    ev = np.asarray([r[4] for r in rows], dtype=np.int32)
    depth = np.asarray([r[5] for r in rows], dtype=np.int16)
    gid = np.asarray([r[6] for r in rows], dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    games = np.unique(gid)
    val = set(rng.choice(games, size=max(1, int(len(games) * 0.1)),
                         replace=False).tolist())
    out = args.out if os.path.dirname(args.out) else paths.data(args.out)
    np.savez_compressed(
        out,
        own=pack_boards(own),
        opp=pack_boards(opp),
        last=last,
        ply=ply,
        eval=ev,
        depth=depth,
        game_id=gid,
        is_val=np.isin(gid, list(val)),
        eval_weight=np.ones(len(rows), dtype=np.float32),
        meta=np.array(json.dumps({
            "source": f"RAPFI25 Eval at {args.time_ms} ms, BOARD positions",
            "positions": len(rows),
            "games": int(len(games)),
            "split": "by game",
            "eval_scale": 200.0,
        })),
    )
    print(f"\nsaved -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"positions {len(rows):,} | games {len(games):,} | "
          f"val {int(np.isin(gid, list(val)).sum()):,} | errors {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
