# Independent quick LDM document witness

PASS: 195 independent evidence-consistency checks passed. One documented invocation ran once from `/data/code/ouyanghao/oyh/textop-reproduction-20261024` and returned actual supervisor exit 0 and actual child exit 0, with the child reaped, no timeout, no cleanup error, no signals required, and output writers closed.

```text
python3 scripts/supervise_quick_ldm.py --mode witness --run ldm_kernel_document_v1
```

The command was launched through a detached durable Python driver, preserving the original stdout/stderr bytes and its actual `subprocess.run` return code. Start: 2026-10-26T19:35:43.079078+08:00; finish: 2026-10-26T19:35:57.550204+08:00. Driver/supervisor/child PIDs: 4047432/4047433/4047440; all were absent after completion. Actual supervised elapsed time was 14.349130s; kernel receipt elapsed time 11.944469s.

## Actual numerical evidence

- Real accepted 512-example batches, four sequential primitives each with history `(2,69)`, future `(8,69)`, and CLIP512 text. Four genuine AdamW updates, zero skipped updates, all four training gradients finite. Update clock 1→2→3→4 comes from wrapping the actual optimizer step. Training losses: 0.8918305039, 0.8067612648, 0.7928245068, 0.7906935215.
- The denoiser and Adam state were serialized into 223,730,007 memory bytes. After deliberately perturbing a model parameter, strict recursive type/dtype/shape/value checks confirmed exact restored model and Adam state. Python, NumPy, Torch CPU, every CUDA RNG state, and separate train/validation generator states matched exactly. The next real training batch replayed exactly. Four pending validation primitives completed with zero optimizer updates.
- Frozen VAE module SHA `044521e6d8c3cedf7cfe14a544a501de97fe5effac8eac387523c24ff89a6112` matches the genuinely trained VAE2000 receipt. Full VAE checkpoint and already existing shared HOME cache each remain 513,523,901 bytes, SHA `5109f593309349504d52677997bcdd21de3d27e045b9154be8c06278182b1219`; no duplicate cache was created.
- The real five-step cosine x0 sampler ran CFG5 for 12 validation histories × four autoregressive primitives. Posterior future features were not input to sampling; generated history was used after the first primitive. Decoded samples were finite. Normalized sampling Huber was `0.3134639114141464`. This is a custom numerical diagnostic, with no semantic or physical score claim.
- Author_v2 and this fresh document run produced identical seeded train records, validation records and sampling diagnostic. Memory serialization byte hashes can differ across processes; the exact within-run state comparison is the relevant restoration check.
- No persistent LDM diagnostic checkpoint was written, no diagnostic weights were reused for training, EMA was disabled, and no SDK or physics steps ran. The run directory contains only the original receipt and metrics. Shared checkpoint directory contents are unchanged: one full `vae.pth` and the existing `vae_src.log`.

## Preflight and immutable inputs

Read the run-experiment skill, its compute environment contract, local SERVER_NOTES ledger, CURRENT QUICK_LDM_ENVIRONMENT, fixed runtime spec, and all three LDM sources before dispatch. No applicable AGENTS.md was found in the workspace ancestors or work tree.

Preflight independently verified author_v2 actual wrapper/child exit0, closed writers and receipt/process/resources SHA chain. It reverified VAE2000 actual 2000 optimizer updates/zero skips, reconstruction gate pass, complete file-resume model/Adam/all-RNG proof, full checkpoint SHA and shared-cache SHA. Accepted 42-source corpus retains 31 TRAIN/11 VAL sources, 174 annotation pairs and 74 CLIP texts; all eight actual corpus artifacts were rehashed. TRAIN-only normalization, source/pose separation and mirror exclusion remain declared and SHA bound.

GPU1 free memory was 40,726 MiB, above the explicitly authorized 20 GiB shared-GPU threshold. CPU0–3 affinity was actually observed after launch; the profile fixes Torch allocation fraction0.65. Before launch RAM was 138,246,918,144 bytes; data disk free 6,964,281,344 bytes; HOME checkpoint disk free 6,094,295,040 bytes against required 3,942,645,760 bytes. Persistent ext4 ownership/no-symlink/writable/budget checks passed. Common lock was free before and after. Paused old57D PID3237858 retained exactly the same PID/start-time/paused-state process row. No other job was signaled.

All 31 collected source/config files retain exact pre/post bytes, including all 19 frozen runtime/control files, new LDM sources, active spec and environment binding. Original corpus/data-acceptance/VAE/author receipts and evidence also retain exact bytes. Every collected file was independently rehashed after local transfer. The train gate remained false and witness metadata was not edited by this agent.

## Doc versus reality and limits

1. CURRENT document says author_v2 dispatched/pending verification; preflight independently confirmed it already had genuine wrapper/child exit0. This is status wording only; the required invocation and profile matched.
2. Short witness (14.349s supervised) ended before the 15-second periodic resource sample. Original samples capture before-launch, after-launch CPU affinity, and after-exit; no live nvidia per-process GPU peak is independently sampled. Actual Torch peak is recorded by the CUDA runtime.
3. Corpus construction receipt retains BUILT_LOADER_PENDING/quick_training_ready=false; separate SHA-bound data-acceptance receipt records accepted corpus/loader. This is preserved layered state, with no claim of full-paper dataset acceptance.

The documented invocation worked verbatim with no repair, reinstall, source/spec/data change, training, additional diagnostic, repeated invocation, author rerun, threshold change, or blind retry. This certifies the documented quick-runtime numerical path only. Parent owns any subsequent witness binding/train-gate activation; this agent has not launched LDM3000.

## Evidence hashes

| Artifact | SHA256 |
|---|---|
| Witness receipt | `d7819dba4d7ebd99c0a9c0b49359268a3cfbfaf8046468125a5c30c2553d42c3` |
| Supervisor exit proof | `8185fa32cdff9c89f631db1385e2e679027d7af561b9c47dd3aba4b18ec2c1cd` |
| Process exit proof | `c73efa2e6864533f4cd1eb51b802085b2d79ba9a68351bc1b3f1d622ea4b9581` |
| Original resource samples | `7c538d86519a5adb724f7e14c1445b50cdf7d8ce046bddee55dc4c27aa16fab0` |
| Durable actual subprocess proof | `0a99398399a2c3b05768017fb7c700d147408c1edf0b765b5a58b3fdbdcdb5a3` |
| Active spec pre/post | `0a2d838bfb9e5b87af300f59e1a1842cb2601ef25477719cc1f92f1b93c1137d` |
| Entrypoint | `68551ebd93f43353fc6c6b9eb45701f13f3c478de1c393750c9755df5dfc962a` |
| Quick LDM runtime | `558d3b91e511e12658c309037ea476d01bdf76c73ce323caf2fc909e269043a3` |
| Supervisor source | `5cff6053c2d2c586b11335d539da6c61c6cad8884dfd5b7332bc9ee739ab4a04` |
| In-memory serialized state | `e9fb0ad8b5776a3cf17d077c12f05ac2570568e8a9de45cf634821e208475d34` |

`before/` and `after/` preserve original actual source/config bytes and receipts; `after/run_originals/` preserves original kernel log, metrics, resource samples, process proof, exit receipt and durable wrapper proof. `supervisor.stdout.bin` and `supervisor.stderr.bin` preserve exact outer process bytes. `SUMMARY.json` enumerates every check and source hash. Local collection/launch/driver source is retained so the evidence path is reviewable.
