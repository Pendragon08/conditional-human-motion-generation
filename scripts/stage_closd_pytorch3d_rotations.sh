#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG=v0.3.0
ARCHIVE="$ROOT/cache/closd-r3-tmp/pytorch3d-${TAG}.tar.gz"
TARGET="$ROOT/envs/closd-r3/site-packages/pytorch3d/transforms"
mkdir -p "$(dirname "$ARCHIVE")" "$TARGET"
if [[ ! -s "$ARCHIVE" ]]; then
  curl --fail --location --connect-timeout 10 --max-time 120 \
    --output "$ARCHIVE" "https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/refs/tags/$TAG"
fi
tar -xOf "$ARCHIVE" "pytorch3d-0.3.0/pytorch3d/transforms/rotation_conversions.py" > "$TARGET/rotation_conversions.py"
tar -xOf "$ARCHIVE" "pytorch3d-0.3.0/LICENSE" > "$ROOT/envs/closd-r3/PYTORCH3D-LICENSE"
cat > "$ROOT/envs/closd-r3/site-packages/pytorch3d/__init__.py" <<'PY'
# CLoSD uses only these pure-Python rotation conversions from PyTorch3D v0.3.0.
PY
cat > "$TARGET/__init__.py" <<'PY'
from .rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion
PY
sha256sum "$TARGET/rotation_conversions.py" "$ROOT/envs/closd-r3/PYTORCH3D-LICENSE"
