"""Publish a frozen model and the Gomocup dataset to the Hugging Face Hub.

Requires one thing this repository cannot provide: an authenticated session.

    hf auth login          # or: huggingface-cli login / HF_TOKEN=...

That opens a browser; the token stays on your machine and is never seen by this
script. Then:

    python code/push_hf.py --user w2Tao --dry-run
    python code/push_hf.py --user w2Tao

The default publishes the historical `baseline` and both datasets. The final
model of the current architecture is published separately:

    python code/push_hf.py --user w2Tao --model-id v1 --model-repo gomoku-v1 \\
        --model-card hf/v1/README.md --skip-dataset

Layout it uploads, built from the repository itself:

    <model-repo>      model repo      ckpt/<model-id>/model.pt + manifest.json
                                      + <model-card> as the model card
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(*parts):
    return os.path.join(ROOT, *parts)


DATASET_FILES = [
    (_path("data", "gomocup15.npz"), "gomocup15.npz"),
    (_path("data", "teacher15.npz"), "teacher15.npz"),
    (_path("hf", "dataset", "README.md"), "README.md"),
]


def model_files(model_id, card):
    return [
        (_path("ckpt", model_id, "model.pt"), "model.pt"),
        (_path("ckpt", model_id, "manifest.json"), "manifest.json"),
        (_path(card), "README.md"),
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
        staged.append((target, dst, os.path.getsize(target) / 1e6))
    return staged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Hugging Face username or org")
    ap.add_argument("--model-id", default="baseline",
                    help="frozen directory under ckpt/")
    ap.add_argument("--model-repo", default="gomoku-baseline")
    ap.add_argument("--model-card", default="hf/model/README.md")
    ap.add_argument("--dataset-repo", default="gomoku-15x15")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
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
            sys.exit(f"not authenticated ({type(e).__name__}). "
                     "Run `hf auth login` or set HF_TOKEN first.")

    model_stage = _path("hf", "_stage_model")
    data_stage = _path("hf", "_stage_dataset")
    mfiles = [] if args.skip_model else stage(
        model_files(args.model_id, args.model_card), model_stage)
    dfiles = [] if args.skip_dataset else stage(DATASET_FILES, data_stage)

    for label, files in (("model", mfiles), ("dataset", dfiles)):
        print(f"\n{label}:")
        for _, name, mb in files:
            print(f"  {name:<20}{mb:7.2f} MB")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    api = HfApi()
    repos = []
    if mfiles:
        repos.append((f"{args.user}/{args.model_repo}", "model", mfiles))
    if dfiles:
        repos.append((f"{args.user}/{args.dataset_repo}", "dataset", dfiles))

    for repo_id, kind, files in repos:
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
