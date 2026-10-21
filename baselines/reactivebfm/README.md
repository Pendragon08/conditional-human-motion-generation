# ReactiveBFM reproduction: bounded training milestone

This is a compatibility pilot, **not a reproduction of the paper's benchmark results**.
The official full-size text-conditioned flow planner completed 101 AdamW updates
on the official Kungfu training split. Physical closed-loop control is not yet evaluated.

- [Chinese experiment report](reports/ReactiveBFM复现实验报告.md)
- [Exact run receipt](results/pilot_v1/run_receipt.json)
- [Per-step metrics](results/pilot_v1/training_metrics.jsonl)
- [Official data manifest](configs/official_data_files.json), [download receipts](results/acquisition_v1/acquisition.json)
- [Numeric data audit](results/data_audit_v1.json)
- [Independent environment witness](results/document_witness_v1/witness.md)
- [Environment specification](configs/env_spec.json)
- [Integrity audit: provisional WARN](results/integrity_audit_v1/EXPERIMENT_AUDIT.md)

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
`D:/具身智能/ReactiveBFM复现_20261021/data/`. Kungfu is also staged in the server's
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

Full mixed-data training (official recipe reaches 1M steps), the unreleased
goal/reaching data and evaluation suite, TensorRT compilation, and physical
MuJoCo closed-loop evaluation remain outstanding. No real-robot result is claimed.
The CPU tracker probe checks model loading and qpos-policy computation only.
All original datasets, code, checkpoints and experiment results remain preserved.
