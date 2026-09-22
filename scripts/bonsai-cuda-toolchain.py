#!/usr/bin/env python3
"""Install a pinned, user-local Windows CUDA compiler for Bonsai benchmarks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "downloads"
    cache.mkdir(exist_ok=True)
    base = "https://developer.download.nvidia.com/compute/cuda/redist/"
    with urllib.request.urlopen(base + "redistrib_12.9.1.json", timeout=60) as response:
        manifest = json.load(response)
    for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcublas"):
        item = manifest[name]["windows-x86_64"]
        archive = cache / Path(item["relative_path"]).name
        if not archive.exists():
            print(f"Downloading {name}: {int(item['size']) / 1e6:.1f} MB", flush=True)
            temporary = archive.with_suffix(".partial")
            with urllib.request.urlopen(base + item["relative_path"], timeout=120) as source:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            temporary.replace(archive)
        if digest(archive) != item["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        print(f"Verified {name}; installing", flush=True)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                parts = Path(member.filename).parts[1:]
                if not parts or member.is_dir():
                    continue
                relative = Path(*parts)
                if len(parts) == 1 and parts[0].lower().startswith(("license", "version")):
                    relative = Path("metadata") / name / relative
                target = (root / relative).resolve()
                if not target.is_relative_to(root):
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
                data = package.read(member)
                if target.exists():
                    if target.read_bytes() != data:
                        raise RuntimeError(f"Conflicting installed file: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    print(f"CUDA 12.9.1 ready: {root}", flush=True)


if __name__ == "__main__":
    main()
