"""Download Gomocup engines for use as an external yardstick.

The engines live at download.gomocup.com and are not in git; this fetches and
unpacks them into `data/engines/<NAME>/`. Re-runnable: a cached zip and an
extracted directory are both skipped.

    python code/fetch_engines.py --list
    python code/fetch_engines.py EMBRYO26
    python code/fetch_engines.py --all

Downloads are chunked with retries. A single-shot read of the 42 MB RAPFI archive
died with `IncompleteRead` partway through, and a truncated zip fails much later,
inside the engine, in a way that looks like a protocol bug.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE = "http://download.gomocup.com/ai"
ROOT = os.path.join("data", "engines")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")

# prefer: substring of the executable to use. Engines ship rule-specific builds --
# EMBRYO has _r/_c5/_s/_f variants and only _f is freestyle -- and picking the
# wrong one silently plays a different game.
ENGINES = {
    "RAPFI25": dict(prefer="rapfi_avx2", size=42.3,
                    note="2026 champion. NNUE; needs a time limit or it deepens forever"),
    "EMBRYO26": dict(prefer="_f", size=6.3,
                     note="2026 6th place, classical alpha-beta; use the _f build"),
    "MONAS": dict(prefer="Monas_x64", size=0.6,
                  note="2026; BUG: emits x=0 along the left edge"),
    "TITO14": dict(prefer="Tito", size=0.04,
                   note="2014, small and fast; weakest entry here"),
}


def download(name, dest_zip):
    url = f"{BASE}/{name}.zip"
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                want = int(resp.headers.get("Content-Length") or 0)
                blob = bytearray()
                while True:
                    chunk = resp.read(1 << 18)
                    if not chunk:
                        break
                    blob += chunk
            if want and len(blob) != want:
                print(f"    short read {len(blob)}/{want}, retrying")
                time.sleep(1.0)
                continue
            with open(dest_zip, "wb") as f:
                f.write(blob)
            return len(blob)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{name}: HTTP {e.code} for {url}")
        except Exception as e:
            print(f"    {type(e).__name__}: {e}")
            time.sleep(1.0)
    raise SystemExit(f"{name}: download failed after 3 attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="engines to fetch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or (not args.names and not args.all):
        print(f"{'engine':<12}{'zip':>8}  build / note")
        print("-" * 72)
        for name, meta in ENGINE().items():
            print(f"{name:<12}{meta['size']:>6.2f}M  "
                  f"{meta['prefer']:<10} {meta['note']}")
        print("\nusage: python code/fetch_engines.py EMBRYO26   (or --all)")
        return 0

    names = list(ENGINES) if args.all else args.names
    os.makedirs(ROOT, exist_ok=True)
    for name in names:
        if name not in ENGINES:
            print(f"  {name}: unknown (see --list)")
            continue
        dest = os.path.join(ROOT, name)
        zpath = os.path.join(ROOT, f"{name}.zip")
        if os.path.isdir(dest) and any(
                f.lower().endswith(".exe") for _, _, fs in os.walk(dest) for f in fs):
            print(f"  {name:<10} already extracted")
            continue
        if not os.path.isfile(zpath):
            print(f"  {name:<10} downloading {ENGINES[name]['size']:.1f} MB")
            size = download(name, zpath)
            print(f"  {name:<10} {size/1e6:.2f} MB")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        exes = [f for _, _, fs in os.walk(dest) for f in fs
                if f.lower().endswith(".exe")]
        print(f"  {name:<10} extracted, {len(exes)} executables")
    return 0


def ENGINE():
    return ENGINES


if __name__ == "__main__":
    sys.exit(main())
