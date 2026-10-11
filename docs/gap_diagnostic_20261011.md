# Literature-grounded reference-quality diagnostic, 2026-10-11

This is a diagnostic of candidate gaps, not a new method or a replication of Morph/PARC/CLAIMS. No VLM/API, policy training, package install or native source change.

## Frozen environment and invocation

Read `.aris/compute/closd-baseline-20261010.md` and `configs/env-closd-baseline-20261010.json`. Reuse unchanged runtime and source hash. Server: `ouyanghao@10.147.17.32`, local key `C:/Users/Pende/.ssh/id_ed25519`; root `/data/code/ouyanghao/oyh/conditional-human-motion-generation`.

Before invocation inspect disk/GPU/processes and the baseline pipeline state. GPU0's existing idle viewer (<1500MiB, <5% utilization) is a documented shared exception; do not stop it. Leave GPU1 untouched. If another training/diagnostic owns GPU0, do not launch. Require >10GiB data free and >40GiB GPU free. The baseline pipeline must be COMPLETE.

Fresh document-following witness executes exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/closd_gap_diagnostic_20261011.sh --check"
```

Expected exit0; `CLOSD_R3_GPU_WITNESS` from existing witness and `GAP_DIAGNOSTIC_PASS` from the four-environment first-attempt recording. Check `results/gap_diagnostic_20261011/v3/witness/receipt.json`: finite shapes, four retained first attempts, matching official commit, no model/policy updates. Check that reference varies across time for at least one active motion and is taken from `ref_body_pos` BEFORE policy action. The inherited reward's separate dataset cache is not the generated reference. Retain failed attempt files if invocation diverges; do not delete or silently rerun.

Alignment correction v3: v2's reward cache was constant/mismatched because the inherited reward reads a separate dataset cache. Its first run is INVALID for tracking conclusions. Capture generated `ref_body_pos` used in the latest observation BEFORE applying the policy action, and actual state after that SAME step; preserve the observation's `frame_idx` for boundary tagging. Do not compare with the newly replanned next observation. This is the operational pre-action-reference/post-action-state measurement, not a theoretical zero-latency error. Parent-folder v2 results are excluded and preserved. Source behavior and all predeclared thresholds are unchanged.

After witness passes the parent launches `--run` in tmux `gap_diagnostic_20261011_v3` and checks `v3/run.state`. Three seeds (11,12,13), 64 independent environments per seed, one FIRST episode each. No conditional resampling of failures. Maximum 315 steps at 30Hz per process, max30min per seed. Expected raw data <100MiB per seed. Final sentinel `COLLECTION_COMPLETE 3x64 first attempts`.

## Predeclared questions

H1: Does completing without official early termination also imply small reference/execution errors and satisfactory foot-contact proxies? Compare completion-only acceptance with diagnostic quality gates. These gates are operational thresholds, NOT ground-truth physical validity labels.

H2: Are replanning boundaries associated with increased kinematic reference speed/change or tracking errors relative to internal frames? Record paired reference and executed positions and matched boundary indices. Observational association is not a causal proof.

Discard no first attempts. Exclude first30frames only from quality statistics to separate initial HumanML conditioning from continued generation; include all frames for termination counts. Main descriptive thresholds: mean g-MPJPE <=100mm; p95 frame g-MPJPE <=200mm; contacted foot-point speed proxy exceedance <=5% at speed0.5m/s and net-force10N. Sensitivity grid: mean50/100/150mm; tail100/200/300mm; force5/10/20N; speed0.25/0.5/1.0m/s. Do not call these universal standards or change them after seeing results.

For force-contact proxies use ankle/toe rigid-body centers and world-XY motion during persistent contact. Rotation/pivot can move centers without surface slip; report as proxy and inspect examples. Do not infer intended contact labels, force correctness, text success, torque limits, or robot executability from them.

Report per-seed results and pooled first-attempt counts. No formal method-versus-method superiority, downstream training benefit, robot deployment, or paper novelty claim can be established by this pilot.

## Completion and measured conclusions

2026-10-11 v3 witness passed exactly once (NPZ SHA2567072a0303d31c82af973043fba336e4f247b1781e1055578f592204b27ec54f3). Official source clean, canonical environment unchanged. Formal run finished with `COLLECTION_COMPLETE 3x64 first attempts`. Each seed retained64 first attempts; 299 steps at30Hz; collection runtime40.45/41.36/41.52 seconds (analysis and transfers excluded). Raw records total31,924,710 bytes.

Non-termination counts59/57/58 (174/192); frozen tracking gate accepts54/52/53 (159/192); combined contact-proxy gate accepts29/29/27 (85/192). Fifteen non-terminated clips fail the mean/tail tracking gate. This supports H1 only at the chosen operational thresholds; it does not attribute failure to the generator or certify data correctness. One early failure has insufficient post-prefix data and remains in termination counts.

H2 tracking worsening NOT supported: reference boundary-minus-internal speed+2.2016m/s; actual speed+0.0583m/s; tracking error-34.64mm. Online replanning reconditions and aligns reference plans to executed states. A plan change is not an actual physical discontinuity.

Contact-proxy sensitivity spans63--109 accepted attempts over the27 frozen threshold combinations. True surface slip is not measured; do not call85 a true physical-quality acceptance count. Missing-scene prompts (e.g. stairs/railing) are outside this flat-ground T2M setting and must not be pooled into a claim of generic data defects.

Analysis scripts: `analyze_gap_diagnostic_20261011.py` verifies raw hashes and v3 alignment; `audit_gap_diagnostic_20261011.py` records hypothesis status and all15 non-terminated tracking rejects. Keep v2 marked INVALID, excluded. Raw licensed arrays stay local/server and are gitignored; share derived metrics, receipts and protocol only.
