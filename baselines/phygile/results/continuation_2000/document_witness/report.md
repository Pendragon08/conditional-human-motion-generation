# Independent document-following resume witness

Result: PASS. On 2026-10-17 the fresh witness agent read only the run-experiment skill, docs/environment.md and docs/continuation_2000.md before execution, and ran the documented one-update invocation verbatim on the authorized original server. No full continuation was launched, no code/docs/shared installation was edited, and no other job was stopped.

```sh
cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017
bash scripts/train_extended_2000.sh 1 32 resume_validation2000
```

SSH: ouyanghao@10.147.17.32; BatchMode=yes; ConnectTimeout=15; authorized existing key. Environment spec: fa5883c8. GPU preflight: GPU 0 RTX A6000, 2968 / 49140 MiB and 7% utilization; existing allocations retained under the environment ledger.

Exit code: 0. Exactly one new learning iteration, 500/501, completed (768 timesteps). PHYGILE_VERIFIED_RESUME reported policy/optimizer/normalizer equality to the parent, 500 previous optimizer updates, first new index 500, one requested additional update, and one active expert. Optimizer and algorithm learning rate agreed after synchronization at 1.5000000000000005e-05.

Read-only artifact inspection passed (exit 0). Output directory: /data/code/ouyanghao/oyh/phygile-reproduction-20261017/logs/rsl_rl/phygile_moe_pilot/2026-10-17_11-23-49_resume_validation2000. It contains model_500.pt (67,349,178 bytes) and resume_provenance.json. The only model_*.pt file in this run is model_500.pt.

Parent checkpoint: /data/code/ouyanghao/oyh/phygile-reproduction-20261017/logs/rsl_rl/phygile_moe_pilot/2026-10-17_09-16-39_train20_nostate_seed42/model_499.pt. Recorded parent SHA-256: afad21ce7192dc4ec10e04fd2bc7684e9278f901dbffcb053cf45b8b04bda162.

No divergence from the documented success criteria was observed. The runtime also logged DriverShaderCacheManager::init() called without a shutdown(), a torch.load FutureWarning, and skipped git-repository discovery; these did not prevent successful exit and checkpoint output. This bounded witness establishes invocation and checkpoint-resume behavior, not convergence or bit-exact trajectory continuation; simulator, RNG, rollout buffer and sampler state restart as documented.

Evidence: stdout.log, stderr.log, exit_code.txt, invocation.sh, started_utc.txt, finished_utc.txt; preflight.stdout.log / preflight.stderr.log / preflight.exit_code.txt; artifact_inspection.stdout.json / artifact_inspection.stderr.log / artifact_inspection.exit_code.txt.
