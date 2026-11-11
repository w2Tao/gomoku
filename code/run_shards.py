"""Launch N shard workers concurrently and wait for all of them.

`Start-Process` from the harness shell dies immediately here (no error, no output
file), so the fan-out is driven from Python instead -- subprocesses with file
redirection, which is the same mechanism the engine driver already relies on.
"""
from __future__ import annotations

import subprocess
import sys
import time

SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2040
TIME_MS = sys.argv[3] if len(sys.argv) > 3 else "300"
TAG = sys.argv[4] if len(sys.argv) > 4 else "dist2"
OPENING = sys.argv[5] if len(sys.argv) > 5 else None

procs = []
for i in range(SHARDS):
    log = open(f"{TAG}_s{i}.log", "w", encoding="utf-8")
    extra = ["--opening", OPENING] if OPENING else []
    p = subprocess.Popen(
        [sys.executable, "-u", "code/build_distill_dataset.py",
         "--games", str(GAMES), "--shards", str(SHARDS), "--shard", str(i),
         "--time-ms", TIME_MS, *extra,
         "--out", f"{TAG}_s{i}.npz"],
        stdout=log, stderr=subprocess.STDOUT)
    procs.append((i, p, log))
print(f"launched {SHARDS} shards, {GAMES} games total", flush=True)

t0 = time.time()
for i, p, log in procs:
    p.wait()
    log.close()

dt = time.time() - t0
print(f"all shards finished in {dt/60:.1f} min")
for i in range(SHARDS):
    with open(f"{TAG}_s{i}.log", encoding="utf-8") as f:
        tail = [ln.rstrip() for ln in f if ln.strip()][-2:]
    print(f"  shard {i}: {' | '.join(tail)}")
