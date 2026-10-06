# Ground-Pick completed-model evaluation recovery — 2026-10-02

Training used the original one-reference Ground-Pick task,8192 environments,2000 PPO iterations,seed42. Run `2026-10-02_04-54-43-Ground-Pick-a` reached1999/2000 and returned normally before evaluation. Final model SHA256 `0b853c6700957045cda3cb768f7d4ffc2ca7cfa67e79fc5fe33d8574f602e699`. Training log SHA256 `88abdc657ce70c7c7e6ab3b8672a2ca5632efbb6d1273482d4a4dde7d7c6ba68`.

The default official video evaluation aborted at simulation initialization with `malloc(): invalid size (unsorted)`, wrapper exit1; no policy evaluation success metric. Original failed log SHA256 `4af4b08f7f5a83c84e1c7edaea0eb2fd183704dfc11b5e00a55afa95a67c0380`. Specific native-memory-corruption root cause remains unconfirmed. No zero score, retraining or reduced budget is authorized by this failure.

The existing audited generic recovery script may perform one supported official `play_eval.py --disable_fabric` evaluation, using USD scene I/O. This retains the same final model,original physics/reward/reference,1000 environments,500-frame evaluation and video. It is an explicitly recorded runtime adaptation, not identical-default-runtime replication. No package changes. A second failure stops; do not retry indefinitely.

Fresh independent read-only validation from a separate shell process, after reading this task-specific document:

```text
python3 scripts/r3_dreamcontrol_completed_eval_recovery.py --task Ground-Pick --model-sha 0b853c6700957045cda3cb768f7d4ffc2ca7cfa67e79fc5fe33d8574f602e699 --check
```

Run from `/data/code/ouyanghao/oyh/conditional-human-motion-generation`. Expected exit0 and `COMPLETED_EVAL_RECOVERY_PREFLIGHT_PASS task=Ground-Pick final1999 hash_verified envs1000 frames500`. The checker acquires the single original queue lock, verifies existing final model/native error/blocked state/previous user-confirmed actual-state display gate and the unused adapted evaluation log. This is input validation, not an evaluation result. Independently verify the original tmux session is absent before launch.

After the checker passes, launch this same command without `--check` once in the original unique tmux session `r3_dreamcontrol_remaining_20261001`, logging to `artifacts/monitoring/r3_Ground-Pick_usd_eval_recovery_20261002_queue.log`. The generic recovery first backs up the final model, original logs and blocked states under `artifacts/recovery/Ground-Pick_20261002_final1999`, checks hashes and then evaluates. It resumes the unchanged original scheduler only after normal exit,official success metric and nonempty positive-duration video; completed Ground-Pick is then skipped and Ground-Pick-Top begins. No active training is interrupted.

Final acceptance additionally verifies501frames/10.02seconds,video/model hashes and the original evaluation failure record. Preserve scalar/config artifacts and upload only scripts, reports, metrics, configuration and permissible logs/videos to the new private repository. CLoSD formal3072 remains blocked pending the already-requested new actual-scene confirmation; do not restart its preflight or infer confirmation from this recovery.
