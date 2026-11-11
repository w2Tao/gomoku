"""Slice a dataset by ply, for the opening/in-game model split.

    python code/slice_by_ply.py IN.npz OUT.npz CUT [max|min]

`max` keeps ply <= CUT (the opening), `min` keeps ply > CUT (in-game).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "code")

import paths                       # noqa: E402
from dataset import load           # noqa: E402


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: slice_by_ply.py IN.npz OUT.npz CUT [max|min]")
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


if __name__ == "__main__":
    main()
