# ReactiveBFM reproduction: full training in progress

The full available-data 1M-update schedule is running on two RTX A6000 GPUs.
The simulator closed loop executes actual MuJoCo PD control and real state feedback;
the earlier 101-update pilot model falls. Neither full training completion nor the
paper's benchmark performance is claimed.

- [Current Chinese training and closed-loop report](reports/ReactiveBFM完整训练与闭环进展报告_20261022.md)
- [Live snapshot captured during this delivery](results/full_training_20261021/live_snapshot_20261022_0036/)
- [Full training checkpoints, diagnostics and reports](https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/reactivebfm-full-training-20261022)
- [Simulation execution receipt](results/full_training_20261021/closed_loop_simclock_v1/receipt.json)
- [State feedback consistency](results/full_training_20261021/closed_loop_simclock_v1/feedback_verification.json)
- [Independent full runtime document witness](results/full_training_20261021/document_witness_v1/witness.md)
- [Full training configuration](configs/full_training_v1.json), [backup/evaluation worker](scripts/backup_full_run.py)
- [Archived bounded pilot report](reports/ReactiveBFM复现实验报告.md)
- [Official data manifest](configs/official_data_files.json), [download receipts](results/acquisition_v1/acquisition.json)
- [Numeric data audit](results/data_audit_v1.json)
- [Environment specification](configs/env_spec.json)
- [Earlier pilot integrity audit: provisional WARN](results/integrity_audit_v1/EXPERIMENT_AUDIT.md)

## Official source and attribution

[ReactiveBFM](https://github.com/zjwzcx/ReactiveBFM), pinned commit
`5a79d03cab824eb5917271ba720294e8a1b9d374`.
All 226 files in `upstream/` are byte-for-byte copies; see `upstream_manifest.json`.
The official project declares **CC-BY-NC-SA-4.0**. Keep its attribution, license
and third-party notices; use the upstream README for citation. Supplementary code
here and our pilot model/optimizer release are also provided under CC-BY-NC-SA-4.0.
Third-party data and weights retain their respective terms.

## Data policy and storage

24 official public data files (870,700,765 bytes) are saved in
`D:/具身智能/ReactiveBFM复现_20261021/data/`. All three motion subsets are staged in the server's
project `data/`. Motion arrays, SMPL-X, and third-party BERT/tracker weights are not
redistributed by this repository. Small demo assets bundled in the licensed official source are retained unchanged. Public records contain acquisition URLs, sizes,
SHA256 digests, split summaries and aggregate diagnostics. A size check against
the public listing is not an official signed-checksum verification.

```sh
python scripts/acquire_official_data.py --help
python scripts/audit_official_data.py --help
```

## Re-running this pilot on the existing server

The server-specific overlay reuses existing PyTorch/CUDA and BERT caches.
See `configs/env_spec.json` and `docs/environment.md`; the latter is a frozen
record of the command independently executed, not a command to overwrite a receipt.
`scripts/build_overlay.py` is a reconstruction helper added after the initial build;
it refuses to replace an existing environment and has not been run against it.

Use a new output directory for every run. Reserve at least 3 GiB beyond estimated
checkpoint requirements. Do not stop other users' jobs. The recorded command was:

```sh
ROOT=/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "$ROOT/env/bin/python" "$ROOT/code/scripts/run_bounded_pilot.py" --runtime-root "$ROOT" --output "$ROOT/runs/pilot_v1"
```

Training keeps the official 512-wide, 16-layer, 8-head network and 36D motion
format. It shortens warmup/scheduled forcing to 25/101 steps, batch size 2, enables
cross_prob=0.1, and disables EMA. Those are explicit pilot deviations. The wrapper
adds metric sinks and count-only hooks; it does not change official losses or updates.

Eight deterministic held-out clips, with identical noise draws before/after,
are used for an auxiliary teacher-forced flow objective. This is not FID,
text-motion alignment, tracking error, perturbation success or full test-set evaluation.

## Remaining work

Full mixed-data training is in progress. The unreleased goal/reaching data and
complete paper evaluation suite remain unavailable. TensorRT and real-time
performance are untested. Native PyTorch/MuJoCo closed-loop execution and feedback
wiring have been tested; learned motion/control quality remains to be established
from the actual full-training checkpoints. No real-robot result is claimed.

The local backup worker must remain running with SSH/GitHub access. It only removes
older checkpoint copies after digest-verified publication; latest server copies
remain. Disk exhaustion waits without deleting unverified files; a prolonged DDP
wait can time out after3h. See the current Chinese report for the exact limits.
