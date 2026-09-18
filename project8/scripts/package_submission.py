#!/usr/bin/env python
"""Package only git-tracked project8 files, excluding data/weights by construction."""

import os
import subprocess
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, default=Path("BIO4901_project8_submission.tgz"))
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
env = os.environ.copy()
if Path("/Library/Developer/CommandLineTools").exists():
    env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"
paths = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", "HEAD", "project8"],
    cwd=root,
    env=env,
    text=True,
).splitlines()
for name in paths:
    if "/data/raw/" in name or Path(name).suffix in {
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".joblib",
        ".npz",
    }:
        raise SystemExit("Refusing raw data/weights: " + name)
subprocess.run(
    [
        "git",
        "archive",
        "--format=tar.gz",
        "-o",
        str(a.output.resolve()),
        "HEAD",
        "project8",
    ],
    cwd=root,
    env=env,
    check=True,
)
print(a.output.resolve())
