"""Run a duel at a sample size that can actually decide something.

The distillation questions all came down to sixteen-game and twenty-eight-game
comparisons, where a 14-point gap is inside the noise. Resolving it needs ~150 games
per arm, which serially is over an hour; games are independent, so they shard.

    python code/duel_many.py EMBRYO26 baseline,v1 150 6

Reports each arm's score with a Wilson interval and the two-arm comparison, so the
answer is a yes/no rather than a point estimate.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - m) / d, (c + m) / d


def run_arm(engine, model, games, shards, time_ms):
    procs, files = [], []
    for i in range(shards):
        jf = f"_duel_{engine}_{model}_{i}.json"
        files.append(jf)
        log = open(f"_duel_{engine}_{model}_{i}.log", "w", encoding="utf-8")
        procs.append(subprocess.Popen(
            [sys.executable, "-u", "code/duel_engine.py", engine, model,
             str(games), "--time-ms", str(time_ms),
             "--shards", str(shards), "--shard", str(i), "--json", jf],
            stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    w = l = d = e = 0
    for jf in files:
        try:
            with open(jf, encoding="utf-8") as f:
                r = json.load(f)
            w += r["wins"]; l += r["losses"]
            d += r["draws"]; e += r["errors"]
        except Exception:
            e += 1
    for jf in files:
        for suf in (".json", ".log"):
            try:
                os.remove(jf.replace(".json", suf))
            except OSError:
                pass
    return w, l, d, e


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "EMBRYO26"
    models = (sys.argv[2] if len(sys.argv) > 2 else "baseline,v1").split(",")
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    shards = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    time_ms = sys.argv[5] if len(sys.argv) > 5 else "300"

    print(f"{engine} @ {time_ms} ms | {games} games per arm | "
          f"{shards} shards\n", flush=True)
    res = {}
    for m in models:
        t0 = time.time()
        w, l, d, e = run_arm(engine, m, games, shards, time_ms)
        n = w + l + d
        p, lo, hi = wilson(w, n)
        res[m] = (w, n, p, lo, hi, e)
        print(f"  {m:<10} {w:>3}/{n:<3} = {p:6.1%}  "
              f"95% [{lo:.1%}, {hi:.1%}]  ({e} aborted, "
              f"{(time.time()-t0)/60:.1f} min)", flush=True)

    if len(models) == 2:
        a, b = models
        wa, na, pa, _, _, _ = res[a]
        wb, nb, pb, _, _, _ = res[b]
        if na and nb:
            pooled = (wa + wb) / (na + nb)
            se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
            z = (pb - pa) / se if se > 0 else 0.0
            print(f"\n  {b} minus {a}: {pb-pa:+.1%}  z = {z:+.2f}")
            verdict = ("significant at 95%" if abs(z) >= 1.96
                       else "NOT significant at 95%")
            print(f"  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
