"""Enable an explicit Kick reference directory without changing the default data.

Patch only the two Kick config assignments in the checked-out DreamControl copy.
The original source is retained as a sibling ``.r3_original`` file.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ASSIGNMENT = 'self.ref_motions_path = "../TrajGen/sample/Kick_sim2"'
REPLACEMENT = (
    'self.ref_motions_path = os.environ.get('
    '"DREAMCONTROL_KICK_REF_PATH", "../TrajGen/sample/Kick_sim2")'
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    original = source.with_name(source.name + ".r3_original")
    content = source.read_text(encoding="utf-8")
    if content.count(REPLACEMENT) == 2 and "import os\n" in content:
        print(f"already patched: {source}")
        return
    if content.count(ASSIGNMENT) != 2 or "import os\n" in content:
        raise SystemExit("Unexpected source content; refusing to patch")
    if original.exists():
        raise SystemExit(f"Backup already exists: {original}")
    original.write_text(content, encoding="utf-8")
    updated = content.replace("import numpy as np\n", "import numpy as np\nimport os\n", 1)
    updated = updated.replace(ASSIGNMENT, REPLACEMENT)
    source.write_text(updated, encoding="utf-8")
    print("original_sha256=" + hashlib.sha256(content.encode()).hexdigest())
    print("patched_sha256=" + hashlib.sha256(updated.encode()).hexdigest())
    print(f"patched: {source}")


if __name__ == "__main__":
    main()
