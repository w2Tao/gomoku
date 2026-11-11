"""Launch N eval-dataset shards and merge the result.

Each shard is a separate Python process with one engine instance and one thread;
this avoids the threaded subprocess creation path that stalled under the Windows
sandbox.

    python code/run_eval_shards.py 12 20000 1000 eval20k
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def main():
    shards = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    positions = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    time_ms = sys.argv[3] if len(sys.argv) > 3 else "1000"
    tag = sys.argv[4] if len(sys.argv) > 4 else "eval20k"

    procs, logs = [], []
    for i in range(shards):
        log = open(f"{tag}_s{i}.log", "w", encoding="utf-8")
        logs.append(log)
        p = subprocess.Popen(
            [sys.executable, "-u", "code/build_eval_dataset.py",
             "--positions", str(positions), "--shards", str(shards),
             "--shard", str(i), "--workers", "1", "--time-ms", time_ms,
             "--out", f"{tag}_s{i}.npz"],
            stdout=log, stderr=subprocess.STDOUT)
        procs.append((i, p))
    print(f"launched {shards} eval shards, {positions} positions total",
          flush=True)

    t0 = time.time()
    for _i, p in procs:
        p.wait()
    for log in logs:
        log.close()
    print(f"all shards finished in {(time.time()-t0)/60:.1f} min", flush=True)

    files = [f"data/{tag}_s{i}.npz" for i in range(shards)]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        print("missing shards:", missing)
        return 1
    subprocess.run([sys.executable, "code/merge_eval_shards.py",
                    f"{tag}.npz", *files], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
