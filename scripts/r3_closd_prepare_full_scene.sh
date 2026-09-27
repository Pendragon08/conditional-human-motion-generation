#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/sim-web-visualizer"
REV="20b02f3380b7872bf262ac26859879d1c1385671"
if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/NVlabs/sim-web-visualizer.git "$SRC"
  git -C "$SRC" checkout --detach "$REV"
fi
if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$REV" ]]; then
  echo "Expected sim-web-visualizer revision $REV; inspect checkout before use" >&2
  exit 1
fi
"$ROOT/.venv-pbhc-training/bin/python" -m pip install --no-deps \
  --target "$ROOT/envs/closd-r3/site-packages" \
  meshcat-sim-web-fork==0.4.0 numpy-quaternion==2023.0.4 \
  pycollada==0.9.3 anytree==2.12.1
echo "Full-scene viewer dependencies staged; CLoSD integration remains opt-in."
