# Actual auxiliary pair-filter witness

This warm-reuses server `reactivebfm@5d436660`, recorded in the existing
`work/reactivebfm_reproduction_20261021/repository/baselines/reactivebfm/docs/recovery_provider_ledger_v3.md`.
There is no installation/environment change. Existing pinned ScaleBFM M weights
and its native PyTorch compatibility loader run on CPU with four threads.
GPU visibility is empty. ReactiveBFM training stays active and is checked before
and after. No dataset, checkpoint or experiment evidence is deleted.

The external tracker is an auxiliary data-quality screen, not PhyGile's GMT.
Root forcing is disabled. The robot starts at the reference's first pose and
velocities; official reset converts the quaternion through float32 (checked at
1e-7). Physics is0.005s with four substeps per50Hz control call. Each actual state
is recorded after applying the current reference, before any reset; no auto-reset
or lag fitting occurs. Body poses are recomputed from the same integrated qpos
in separate MjData via mj_kinematics: the live simulator state and solver are not
touched. This avoids mj_step's one-substep-old derived body positions. Full raw
arrays remain in the private server run directory. Historical v1/v2 runs are
retained and superseded for filtering statistics by this corrected v3 observer.

Execute this PowerShell invocation verbatim exactly once. Both local and remote
output directories must be new; existing outputs deliberately fail.

```powershell
$env:PYTHONUTF8='1'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/paper_faithful_reproduction/scripts/witness_scalebfm_pair_filter.py' --output 'D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/scalebfm_pair_filter_document_witness_v2'
```

Expected exit0 and `PAIR_FILTER_DOCUMENT_WITNESS_PASSED`: one real pair/seed42,
123 actual control calls, full-clip mean MPJPE between0.04m and0.07m, unchanged sources,
weights and model parameters, ReactiveBFM still `TRAINING` with nondecreasing
validated updates. This proves one bounded physical run and the invocation;
it does not prove a complete filtered corpus, paper success rate, native
PhyGile GMT training, semantic generation or a trained generator's closed loop.

Three terminal predicates use PhyGile's reported tracking-evaluation thresholds:
pelvisZ error0.3m, torso gravity-projection difference0.8, four named end-effectorZ
errors0.3m. Mean MPJPE below0.1m is an explicit additional reconstruction choice.
The body/name set and the external policy differ from the earlier Unitree pilot,
so the results cannot be used as a matched controller comparison.
