"""Slice a dataset by ply, for the opening/in-game model split. (helper)

    python code/slice_by_ply.py IN.npz OUT.npz max_ply     # ply <= max_ply
    python code/slice_by_ply.py IN.npz OUT.npz min_ply     # ply >= min_ply (flag 2)
"""
import json
import sys

import numpy as np

sys.path.insert(0, "code")

import paths                       # noqa: E402
from dataset import load           # noqa: E402

src, dst, cut = sys.argv[1], sys.argv[2], int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "max"

d = load(src if src.endswith(".npz") and "/" in src else paths.data(src))
ply = d["ply"].astype(int)
m = ply <= cut if mode == "max" else ply > cut

out = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(ply) else v)
       for k, v in d.items() if k != "meta"}
out["meta"] = np.array(json.dumps({
    "source": f"{src} sliced to ply {'<=' if mode == 'max' else '>'} {cut}",
    "positions": int(m.sum()),
}))
np.savez_compressed(paths.data(dst), **out)
print(f"  {src} -> {dst}: {int(m.sum()):,} of {len(ply):,} rows "
      f"(ply {'<=' if mode == 'max' else '>'} {cut})")
print(f"  ply range kept: {ply[m].min() if m.sum() else '-'} .. {ply[m].max() if m.sum() else '-'}")
