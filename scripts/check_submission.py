"""Pre-submission checks for the Megathon repository rules.

Checks tracked files (what will reach `main`), not the working tree:
  * no file over 100 MB (GitHub rejects it) and a warning above 50 MB
  * no virtual environments, datasets, caches or model weights
  * no large blobs anywhere in git history (they block the push too)
  * README.md, LICENSE (MIT) and the deliverables required for a stage exist

Optionally builds code.zip from tracked files only, which excludes virtual
environments, datasets and weights by construction.

    python scripts/check_submission.py --stage final
    python scripts/check_submission.py --stage final --make-zip
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

MB = 1024 * 1024
FORBIDDEN_DIRS = {".venv", "venv", "env", ".conda", "node_modules", "data", "datasets", "artifacts", "results",
                  "content_type_data",
                  "__pycache__", ".pytest_cache", ".gradio", "feedback"}
FORBIDDEN_EXT = {".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npy", ".npz", ".faiss", ".index",
                 ".h5", ".pkl", ".parquet", ".tar", ".gz", ".zip"}
STAGE_FILES = {
    "m1": ["deliverables/M1.pdf"],
    "m2": ["deliverables/M1.pdf", "deliverables/M2.pdf"],
    "final": ["deliverables/M1.pdf", "deliverables/M2.pdf", "deliverables/final_report.pdf"],
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=sorted(STAGE_FILES), default=None)
    parser.add_argument("--make-zip", action="store_true", help="Write code.zip from tracked files")
    args = parser.parse_args()

    root = Path(git("rev-parse", "--show-toplevel").strip())
    tracked = [p for p in git("-C", str(root), "ls-files", "-z").split("\0") if p]
    errors, warnings = [], []

    for rel in tracked:
        path = root / rel
        parts = set(Path(rel).parts[:-1])
        if parts & FORBIDDEN_DIRS or any(d.startswith(("data_", "artifacts_", "results_")) for d in parts):
            errors.append(f"tracked file in excluded directory: {rel}")
        if Path(rel).suffix.lower() in FORBIDDEN_EXT:
            errors.append(f"tracked data/weight file: {rel}")
        if path.exists():
            size = path.stat().st_size
            if size > 100 * MB:
                errors.append(f"{rel} is {size / MB:.1f} MB (GitHub limit is 100 MB)")
            elif size > 50 * MB:
                warnings.append(f"{rel} is {size / MB:.1f} MB (GitHub warns above 50 MB)")

    # Large blobs anywhere in history also make the push fail.
    objects = git("-C", str(root), "rev-list", "--objects", "--all")
    batch = subprocess.run(["git", "-C", str(root), "cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)"],
                           input=objects, capture_output=True, text=True, check=True).stdout
    for line in batch.splitlines():
        kind, size, *name = line.split(" ", 2)
        if kind == "blob" and int(size) > 100 * MB:
            errors.append(f"history contains a {int(size) / MB:.0f} MB blob: {name[0] if name else '?'}")

    if "README.md" not in tracked:
        errors.append("README.md is missing")
    if "LICENSE" not in tracked:
        errors.append("LICENSE is missing")
    elif "MIT License" not in (root / "LICENSE").read_text(encoding="utf-8"):
        errors.append("LICENSE is not the MIT License")
    for required in STAGE_FILES.get(args.stage, []):
        if required not in tracked:
            errors.append(f"{required} is not committed")

    branch = git("-C", str(root), "rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch != "main":
        warnings.append(f"current branch is '{branch}'; only 'main' is evaluated")
    if git("-C", str(root), "status", "--porcelain").strip():
        warnings.append("uncommitted changes in the working tree")

    total = sum((root / p).stat().st_size for p in tracked if (root / p).exists())
    print(f"{len(tracked)} tracked files, {total / MB:.2f} MB")
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR:   {e}")

    if args.make_zip:
        out = root / "code.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in tracked:
                if (root / rel).is_file():
                    zf.write(root / rel, rel)
        print(f"Wrote {out} ({out.stat().st_size / MB:.2f} MB, {len(tracked)} files)")

    print("OK" if not errors else f"{len(errors)} problem(s) found")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
