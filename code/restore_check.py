"""Restore check: can this project be rebuilt from GitHub + Hugging Face alone?

Run from a clean export of the committed tree (git archive HEAD), i.e. with the
working directory set to that export, which has no ignored files. It pulls the published artefacts over plain HTTP (the
hub client insists on symlinks, which the Windows sandbox refuses), then
exercises what a user would actually use: tests, dataset loader, frozen baseline.
"""
import hashlib
import os
import subprocess
import sys
import urllib.request

ARTEFACTS = [
    ("https://huggingface.co/datasets/w2Tao/gomoku-15x15/resolve/main/gomocup15.npz",
     "data/gomocup15.npz"),
    ("https://huggingface.co/datasets/w2Tao/gomoku-15x15/resolve/main/teacher15.npz",
     "data/teacher15.npz"),
    ("https://huggingface.co/w2Tao/gomoku-baseline/resolve/main/model.pt",
     "ckpt/baseline/model.pt"),
]

print("=== 1. what the export contains ===")
have = sorted(os.path.relpath(os.path.join(r, f), ".")
              for r, _, fs in os.walk(".") for f in fs if "__pycache__" not in r)
print(f"  {len(have)} files")
for p in ("data/gomocup15.npz", "ckpt/baseline/model.pt"):
    print(f"  {p:<26}{'present' if os.path.isfile(p) else 'ABSENT (expected)'}")

print("\n=== 2. download the published artefacts ===")
for url, dest in ARTEFACTS:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    blob = urllib.request.urlopen(url, timeout=300).read()
    with open(dest, "wb") as f:
        f.write(blob)
    print(f"  {dest:<26}{len(blob)/1e6:6.2f} MB  "
          f"sha {hashlib.sha256(blob).hexdigest()[:16]}")

print("\n=== 3. test suite ===")
fails = 0
for t in ("tests.py", "test_solvers.py", "test_value_target.py"):
    r = subprocess.run([sys.executable, os.path.join("code", t)],
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    ok = r.returncode == 0
    fails += 0 if ok else 1
    print(f"  {t:<22}{'PASS' if ok else 'FAIL'}  {last}")

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

print("\n=== 5. frozen baseline ===")
import random                                            # noqa: E402
import torch                                             # noqa: E402
import baselines                                         # noqa: E402
from play import PolicyAgent                             # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
net, manifest = baselines.load_baseline("baseline", dev)
agent = PolicyAgent(net, dev)
mv = agent.move(0, 0, -1, random.Random(0))
print(f"  model_type {manifest['model_type']} | params {manifest['parameters']:,}")
print(f"  opening move {mv} -> row {mv // 15} col {mv % 15}  on {dev}")

print("\nresult:", "RESTORABLE" if fails == 0 else "PROBLEMS FOUND")
