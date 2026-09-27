"""Materialize and verify official DreamControl Git LFS dataset pointers."""

import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(os.environ["R3_DREAMCONTROL_ROOT"])
DATA = ROOT / "TrajGen" / "dataset"
API = "https://github.com/GenRobo/DreamControl.git/info/lfs/objects/batch"
MANIFEST = Path(os.environ["R3_LFS_MANIFEST"])


def main():
    pointers = []
    for path in DATA.rglob("*"):
        if not path.is_file():
            continue
        head = path.read_bytes()[:200]
        if not head.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            continue
        parts = dict(line.split(" ", 1) for line in head.decode().splitlines()[1:])
        pointers.append((path, parts["oid"].replace("sha256:", "", 1), int(parts["size"])))
    if not pointers:
        raise RuntimeError("No Git LFS pointers found; inspect dataset state")
    payload = {"operation": "download", "objects": [{"oid": oid, "size": size} for _, oid, size in pointers]}
    req = Request(API, data=json.dumps(payload).encode(), headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
    with urlopen(req, timeout=60) as response:
        objects = {obj["oid"]: obj for obj in json.load(response)["objects"]}
    done = []
    for path, oid, size in pointers:
        obj = objects[oid]
        href = obj["actions"]["download"]["href"]
        headers = obj["actions"]["download"].get("header", {})
        with urlopen(Request(href, headers=headers), timeout=120) as response:
            content = response.read()
        if len(content) != size or hashlib.sha256(content).hexdigest() != oid:
            raise RuntimeError(f"Git LFS verification failed: {path}")
        temp = path.with_name(path.name + ".r3.tmp")
        temp.write_bytes(content)
        temp.replace(path)
        done.append({"path": str(path.relative_to(ROOT)), "size": size, "sha256": oid})
        print(f"Verified {done[-1]['path']} ({size} bytes)", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"source": API, "files": done}, indent=2) + "\n")


if __name__ == "__main__":
    main()
