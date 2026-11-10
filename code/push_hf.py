"""Publish the frozen baseline and the Gomocup dataset to the Hugging Face Hub.

Requires one thing this repository cannot provide: an authenticated session.

    hf auth login          # or: huggingface-cli login

That opens a browser; the token stays on your machine and is never seen by this
script. Then:

    python code/push_hf.py --user <your-hf-username>

Layout it uploads, built from the repository itself:

    gomoku-baseline   model repo      ckpt/baseline/model.pt + manifest.json
                                      + hf/model/README.md as the model card
    gomoku-15x15      dataset repo    data/gomocup15.npz + data/teacher15.npz
                                      + hf/dataset/README.md as the dataset card

Nothing is uploaded unless it exists, and the script refuses to touch a repo it
did not create unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

MODEL_FILES = [
    ("ckpt/baseline/model.pt", "model.pt"),
    ("ckpt/baseline/manifest.json", "manifest.json"),
    ("hf/model/README.md", "README.md"),
]

DATASET_FILES = [
    ("data/gomocup15.npz", "gomocup15.npz"),
    ("data/teacher15.npz", "teacher15.npz"),
    ("hf/dataset/README.md", "README.md"),
]


def stage(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    staged = []
    for src, dst in files:
        if not os.path.isfile(src):
            print(f"  skip (missing): {src}")
            continue
        target = os.path.join(out_dir, dst)
        shutil.copyfile(src, target)
        staged.append((target, dst,
                       os.path.getsize(target) / 1e6))
    return staged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Hugging Face username or org")
    ap.add_argument("--model-repo", default="gomoku-baseline")
    ap.add_argument("--dataset-repo", default="gomoku-15x15")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        try:
            from huggingface_hub import HfApi, whoami
        except ImportError:
            sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
        try:
            me = whoami()
            print(f"authenticated as: {me.get('name')}")
        except Exception as e:
            sys.exit(f"not authenticated ({type(e).__name__}). Run `hf auth login` first.")

    model_stage = os.path.join("hf", "_stage_model")
    data_stage = os.path.join("hf", "_stage_dataset")
    print("\nstaging model repo:")
    mfiles = stage(MODEL_FILES, model_stage)
    print("staging dataset repo:")
    dfiles = stage(DATASET_FILES, data_stage)

    for label, files in (("model", mfiles), ("dataset", dfiles)):
        print(f"\n{label}:")
        for _, name, mb in files:
            print(f"  {name:<20}{mb:7.2f} MB")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    api = HfApi()
    for repo_id, kind, files in (
        (f"{args.user}/{args.model_repo}", "model", mfiles),
        (f"{args.user}/{args.dataset_repo}", "dataset", dfiles),
    ):
        print(f"\n{kind} repo -> {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type=kind,
                        private=args.private, exist_ok=args.overwrite)
        for path, name, _ in files:
            api.upload_file(path_or_fileobj=path, path_in_repo=name,
                            repo_id=repo_id, repo_type=kind)
            print(f"  uploaded {name}")
        print(f"  https://huggingface.co/{kind}s/{repo_id}")


if __name__ == "__main__":
    main()
