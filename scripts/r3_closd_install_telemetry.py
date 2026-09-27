"""Install two opt-in CLoSD telemetry hooks and a serial-loader compatibility switch.

Exact replacements are checked and original files are backed up once. These
changes do not alter DiP, PPO, rewards, or the task distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> str:
    source = path.read_text(encoding="utf-8")
    if new in source:
        return "already_installed"
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one insertion point in {path}, found {count}")
    backup = path.with_suffix(path.suffix + ".r3_upstream")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return "installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closd-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.closd_root.resolve()
    motion_lib = root / "closd/utils/motion_lib_base.py"
    task = root / "closd/env/tasks/closd.py"

    result = replace_once(
        motion_lib,
        "        if flags.debug:\n            num_jobs = 1\n",
        "        if flags.debug or os.environ.get('R3_CLOSD_SERIAL_MOTION_LOADING') == '1':\n"
        "            num_jobs = 1\n",
    )
    print(f"motion_loader: {result}")
    result = replace_once(
        task,
        "        planning_horizon = sample_xyz[:, context_len_30fps-1:context_len_30fps+self.planning_horizon_30fps]  # [x, -z, y]\n\n"
        "        return planning_horizon[:, 0], planning_horizon[:, 1:]\n",
        "        planning_horizon = sample_xyz[:, context_len_30fps-1:context_len_30fps+self.planning_horizon_30fps]  # [x, -z, y]\n"
        "        if os.environ.get('R3_CLOSD_TELEMETRY_UDP'):\n"
        "            try:\n"
        "                from r3_closd_telemetry import emit_plan\n"
        "                emit_plan(self.frame_idx, self.hml_prompts[0], planning_horizon[0])\n"
        "            except Exception:\n"
        "                pass\n\n"
        "        return planning_horizon[:, 0], planning_horizon[:, 1:]\n",
    )
    print(f"plan_hook: {result}")
    result = replace_once(
        task,
        "        self._update_pose_buffer()\n        self.frame_idx += 1\n        if self.save_hml_episodes and self.n_all_episodes < self.max_saved_hml_episodes:\n",
        "        self._update_pose_buffer()\n        self.frame_idx += 1\n"
        "        if os.environ.get('R3_CLOSD_TELEMETRY_UDP') and hasattr(self, 'next_pred_pose'):\n"
        "            try:\n"
        "                from r3_closd_telemetry import emit_step\n"
        "                emit_step(self.frame_idx, self.hml_prompts[0],\n"
        "                          self._rigid_body_pos[0], self.next_pred_pose[0])\n"
        "            except Exception:\n"
        "                pass\n"
        "        if self.save_hml_episodes and self.n_all_episodes < self.max_saved_hml_episodes:\n",
    )
    print(f"step_hook: {result}")


if __name__ == "__main__":
    main()
