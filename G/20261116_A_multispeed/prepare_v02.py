"""Materialize the portable G inputs; asset bytes are pinned to the inherited Git tree."""
from hashlib import sha256
import json
from pathlib import Path
from shutil import copy2, copytree
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def checked(path, identity):
    return path.stat().st_size == identity["bytes"] and sha256(path.read_bytes()).hexdigest() == identity["sha256"]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python prepare_v02.py <new-output-directory>")
    destination = Path(sys.argv[1]).resolve()
    assets = json.loads((HERE/"evidence/v02_assets_from_repo.json").read_text(encoding="utf-8"))
    inputs = json.loads((HERE/"evidence/ready_inputs_sha256.json").read_text(encoding="utf-8"))
    for name, identity in inputs.items():
        if not checked(HERE/name, identity):
            raise ValueError(f"Portable Git input differs from run receipt: {name}")
    for name, identity in assets.items():
        if not checked(REPO/identity["repo_path"], identity):
            raise ValueError(f"Inherited Git asset differs from V02: {name}")
    copytree(HERE/"v02_portable", destination)
    for name, identity in assets.items():
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        copy2(REPO/identity["repo_path"],target)
    print(destination)


if __name__ == "__main__":
    main()
