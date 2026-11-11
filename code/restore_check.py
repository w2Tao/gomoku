"""Does a fresh clone become a working project using only the documented steps?

Run from a clean export of the committed tree (git archive HEAD), i.e. with the
working directory set to that export. Everything gitignored is absent there,
which is exactly the state a new user starts from.

It then follows README's quick start literally -- fetch_data.py, then the tests,
then the baseline -- so a broken instruction shows up as a failure here.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    os.chdir(ROOT)
    print("=== 1. state of the export ===")
    files = sorted(os.path.relpath(os.path.join(r, f), ".")
                   for r, _, fs in os.walk(".") for f in fs
                   if "__pycache__" not in r)
    print(f"  {len(files)} files committed")
    for p in ("data/gomocup15.npz", "data/teacher15.npz",
              "ckpt/baseline/model.pt", "ckpt/v1/model.pt"):
        print(f"  {p:<26}{'present' if os.path.isfile(p) else 'absent (expected)'}")

    print("\n=== 2. README step 1: fetch_data.py ===")
    r = run([sys.executable, os.path.join("code", "fetch_data.py")])
    print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
    if r.returncode != 0:
        print(r.stderr[-500:])
        sys.exit("fetch step failed")

    print("\n=== 3. test suite ===")
    fails = 0
    for t in ("tests.py", "test_solvers.py", "test_value_target.py",
              "test_dataset.py"):
        r = run([sys.executable, os.path.join("code", t)])
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        ok = r.returncode == 0
        if last.upper().startswith("SKIP"):
            state = "SKIP"
        else:
            state = "PASS" if ok else "FAIL"
        fails += 0 if ok else 1
        print(f"  {t:<22}{state}  {last}")

    print("\n=== 4. dataset loader ===")
    sys.path.insert(0, os.path.abspath("code"))
    import paths                                             # noqa: E402
    from dataset import bitboards, load, to_planes            # noqa: E402

    d = load(paths.data("gomocup15.npz"))
    own, opp = bitboards(d)
    x = to_planes(d)
    print(f"  records {len(d['cls']):,} | planes {x.shape}")
    print(f"  colour overlaps {sum(1 for a, b in zip(own, opp) if a & b)} | "
          f"val positions {int(d['is_val'].sum()):,}")

    print("\n=== 5. baselines play ===")
    import random                                            # noqa: E402
    import torch                                             # noqa: E402
    import baselines                                         # noqa: E402
    from play import PolicyAgent                             # noqa: E402

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for bid in ("baseline", "v1"):
        net, manifest = baselines.load_baseline(bid, dev)
        mv = PolicyAgent(net, dev).move(0, 0, -1, random.Random(0))
        print(f"  {bid}: {manifest['model_type']} | "
              f"{manifest['parameters']:,} params | "
              f"opening move {mv} -> row {mv // 15} col {mv % 15}  on {dev}")

    print("\nresult:", "ONBOARDING WORKS" if fails == 0 else "PROBLEMS FOUND")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
