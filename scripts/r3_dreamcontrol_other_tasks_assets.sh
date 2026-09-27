#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$ROOT/third_party/DreamControl"
MON="$ROOT/artifacts/monitoring"
STATE="$MON/r3_dreamcontrol_other_tasks_assets.state"
LOG="$MON/r3_dreamcontrol_other_tasks_assets.log"
MANIFEST="$MON/r3_dreamcontrol_other_tasks_assets.json"
mkdir -p "$MON"

[[ "$(git -C "$SOURCE" rev-parse HEAD)" == ec28b508313d6837de52c68247fbc8cf81fe4935 ]] || {
  echo 'BLOCKED upstream_revision_mismatch' > "$STATE"; exit 1;
}
export PATH="$ROOT/cache/git-lfs-3.8.0:$PATH"
[[ -x "$ROOT/cache/git-lfs-3.8.0/git-lfs" ]] || {
  echo 'BLOCKED verified_git_lfs_missing' > "$STATE"; exit 1;
}

names=(Punch_sim2 Jump_sim2 Sit_sim2 Pick_sim2 Bimanual_Pick_sim2
       Ground_Pick_sim2 Ground_Pick_Top_sim2 Button_Press_sim2
       Open_Drawer_sim2 Open_Door_sim2)
include=""
for name in "${names[@]}"; do
  [[ -d "$SOURCE/TrajGen/sample/$name" ]] || {
    echo "BLOCKED sample_dir_missing=$name" > "$STATE"; exit 1;
  }
  include+="${include:+,}TrajGen/sample/$name/**"
done

echo "RUNNING materializing_public_official_lfs task_count=${#names[@]}" > "$STATE"
if ! git -C "$SOURCE" lfs pull --include="$include" --exclude='' > "$LOG" 2>&1; then
  echo "BLOCKED lfs_pull_failed log=$LOG" > "$STATE"
  exit 1
fi

python3 - "$SOURCE" "$MANIFEST" "${names[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path

root, target = Path(sys.argv[1]), Path(sys.argv[2])
result = {}
for name in sys.argv[3:]:
    files = sorted((root / 'TrajGen/sample' / name).glob('*.pkl'))
    if not files:
        raise SystemExit(f'No reference motions: {name}')
    rows = []
    for path in files:
        if path.read_bytes()[:50].startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise SystemExit(f'Unmaterialized pointer: {path}')
        rows.append({'file': path.name, 'size': path.stat().st_size,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result[name] = {'count': len(rows), 'total_bytes': sum(x['size'] for x in rows),
                    'files': rows}
target.write_text(json.dumps({'upstream_commit': 'ec28b508313d6837de52c68247fbc8cf81fe4935',
                              'tasks': result}, indent=2) + '\n')
print('Verified', ', '.join(f'{name}={entry["count"]}' for name, entry in result.items()))
PY
echo "COMPLETE task_count=${#names[@]} manifest=$MANIFEST" > "$STATE"
