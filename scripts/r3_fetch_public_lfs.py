"""Fetch one public Git LFS object, verifying its pointer's size and SHA-256."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/DreamControl"
REPO_URL = "https://github.com/GenRobo/DreamControl.git/info/lfs/objects/batch"


def main() -> None:
    path = (REPO / sys.argv[1]).resolve()
    if os.path.commonpath((str(path), str(REPO.resolve()))) != str(REPO.resolve()):
        raise SystemExit("Target must remain within DreamControl repository")
    pointer = path.read_text(encoding="utf-8")
    assert pointer.startswith("version https://git-lfs.github.com/spec/v1\n")
    fields = dict(line.split(" ", 1) for line in pointer.splitlines()[1:])
    oid = fields["oid"].split(":", 1)[1]
    size = int(fields["size"])
    body = json.dumps(
        {"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}]}
    ).encode()
    req = urllib.request.Request(
        REPO_URL,
        data=body,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "git-lfs/3.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        info = json.load(response)
    item = info["objects"][0]
    if "error" in item:
        raise RuntimeError(item["error"])
    action = item["actions"]["download"]
    request = urllib.request.Request(action["href"], headers=action.get("header", {}))
    temp = path.with_suffix(path.suffix + ".r3-download")
    digest = hashlib.sha256()
    count = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if count != size or digest.hexdigest() != oid:
            raise RuntimeError(f"Git LFS integrity mismatch: {count}/{size} {digest.hexdigest()}/{oid}")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    print(f"R3_LFS_FETCH_OK {path} bytes={count} sha256={oid}")


if __name__ == "__main__":
    main()
