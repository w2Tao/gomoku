"""Download the published model and data into this checkout.

The weights and the datasets are kept out of git (they live on Hugging Face), so a
fresh clone has no `data/` and no `ckpt/baseline/model.pt`. This fetches them and
verifies each file against a pinned SHA-256:

    python code/fetch_data.py

Run it once after cloning, before `python app.py`. It is safe to re-run: anything
already present with the right hash is skipped, and a partial or corrupt file is
replaced. Plain HTTP is used deliberately -- no `huggingface_hub`, no login, and
no symlinks, which is what makes it work on a locked-down machine.

The four artefacts are the frozen `baseline` (the supervised model), the final
`v1` (the current ValuePolicyNet architecture's final form) and the two datasets.

    python code/fetch_data.py --force     redownload even if hashes match
    python code/fetch_data.py --check     verify only, download nothing

Paths are resolved against the project root, not the current directory, so the
script behaves the same when launched from the parent of the checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = "https://huggingface.co"
MODEL_REPO = "w2Tao/gomoku-baseline"
V1_REPO = "w2Tao/gomoku-v1"
DATA_REPO = "w2Tao/gomoku-15x15"


def _dst(*parts):
    return os.path.join(ROOT, *parts)


# (url, destination, sha256, what it is)
ARTEFACTS = [
    (f"{HUB}/{MODEL_REPO}/resolve/main/model.pt",
     _dst("ckpt", "baseline", "model.pt"),
     "191368ea924ab0d067a77cbcfe4554888e85089342cc18f4ca1ef5fbf109ca1c",
     "frozen baseline weights"),
    (f"{HUB}/{V1_REPO}/resolve/main/model.pt",
     _dst("ckpt", "v1", "model.pt"),
     "444e7d8cfc08b453fe60a6d83591eb8751e23faaf53a21a2177ea607ace2f624",
     "final v1 weights (ValuePolicyNet architecture)"),
    (f"{HUB}/datasets/{DATA_REPO}/resolve/main/gomocup15.npz",
     _dst("data", "gomocup15.npz"),
     "688af62cee7c787ba20ff7e72b81ac9ca65279a09ba8505a255b3898f0498a12",
     "344,989 positions from Gomocup game records (deduplicated)"),
    (f"{HUB}/datasets/{DATA_REPO}/resolve/main/teacher15.npz",
     _dst("data", "teacher15.npz"),
     "c539d36849b4cea9654906ffd8f1ecd346be52b1038788d16bd509f5437ca4da",
     "150,000 positions from heuristic self-play"),
]



def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    """Fetch to a temp file next to the destination, so a failure leaves nothing.

    Transient TLS/EOF failures from the proxy have been observed on large files;
    retry a few times before giving up. A partial temp file is removed either way.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".", suffix=".part")
    os.close(fd)
    try:
        last = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(url, timeout=300) as resp, \
                        open(tmp, "wb") as out:
                    total = int(resp.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        chunk = resp.read(1 << 18)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            sys.stdout.write(f"\r    {pct:5.1f}%  "
                                             f"{done/1e6:5.2f}/{total/1e6:5.2f} MB")
                            sys.stdout.flush()
                sys.stdout.write("\r" + " " * 44 + "\r")
                os.replace(tmp, dest)
                return
            except Exception as exc:                  # noqa: BLE001
                last = exc
                sys.stdout.write("\r" + " " * 44 + "\r")
                print(f"    attempt {attempt}/3 failed: "
                      f"{type(exc).__name__}: {exc}")
                time.sleep(1.0)
        raise last
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="redownload even when the hash already matches")
    ap.add_argument("--check", action="store_true",
                    help="verify what is present, download nothing")
    args = ap.parse_args()

    print("model and data are published on Hugging Face; fetching into this checkout")
    print()

    ready = missing = 0
    for url, dest, want, what in ARTEFACTS:
        have = os.path.isfile(dest)
        ok = have and sha256(dest) == want
        label = os.path.relpath(dest, ROOT)

        if ok and not args.force:
            print(f"  {label:<18} ok        ({what})")
            ready += 1
            continue
        if args.check:
            print(f"  {label:<18} {'WRONG HASH' if have else 'missing'}")
            missing += 1
            continue

        print(f"  {label:<18} fetching  ({what})")
        try:
            download(url, dest)
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {e}")
            print(f"    source: {url}")
            missing += 1
            continue

        got = sha256(dest)
        if got != want:
            print(f"    hash mismatch after download")
            print(f"      expected {want}")
            print(f"      got      {got}")
            os.remove(dest)
            missing += 1
        else:
            print(f"  {label:<18} ok")
            ready += 1

    print()
    if missing:
        print(f"{missing} of {len(ARTEFACTS)} artefacts unusable; project will not run.")
        print("Retry, or download them by hand from the repositories above.")
        return 1
    print(f"all {ready} artefacts present and verified -- ready to run `python app.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
