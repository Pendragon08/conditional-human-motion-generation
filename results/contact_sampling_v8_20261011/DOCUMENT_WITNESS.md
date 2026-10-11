# V8 fresh document witness — PASS

Date: 2026-10-11. Independent document-following agent.
Scope: first-batch measurement interface and documented invocation only.

Read the run-experiment skill, shared compute-env contract, unchanged closd-baseline runtime ledger, and docs/contact_sampling_v8_20261011.md before launch. The project root has no AGENTS.md; the explicit provider ledger supplies the runtime and SSH identity. No environment rebuild, install, source edit, retry, or --run was performed by this witness.

Executed the documented command verbatim exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_sampling_v8_20261011.py --check
```

Exit code: 0. Output:

```text
CONTACT_SAMPLING_BATCH_PASS repeat_a
CONTACT_SAMPLING_DOCUMENT_CHECK_PASS
```

Inspected remote results/contact_sampling_v8_20261011/raw/repeat_a and independently ran read-only hash, array, lineage, and metric recomputation assertions (without calling the mutating certify_cached helper). Verification exited 0 with DOCUMENT_METADATA_AND_NUMERIC_WITNESS_PASS.

- Receipt complete; 12 original cases/attempts retained; all 12 completed with exactly 120 valid frames. Recorded joint_pass=0; this is retained task outcome, not a document failure or efficacy claim.
- Reference frame clock exactly 0..119. No duplicate initial frame.
- initial_reset_gate_pass=true. All seven manual-reset versus first-action maximum differences equal 0: humanoid root, DOF position/velocity, rigid-body position/rotation/linear/angular velocity. The seven state hashes also match exactly.
- All four initial rigid-cache consistency differences equal 0: position, rotation, linear velocity, angular velocity.
- completion_fence_calls=242. Protocol: fetch_results(True) immediately after every native simulate, including reset warmup; no extra simulate or tensor refresh. Source inspection confirms a forwarding Gym proxy invokes native.simulate once followed immediately by native.fetch_results(sim, True).
- Effective SimParams are recorded: dt=0.01666666753590107, substeps=2, use_gpu_pipeline=true, contact_collection=2, physx_use_gpu=true.
- contact_mutation_during_reward_max_N=0; body_coordinate_roundtrip_max_m=0; initial_actual_reference_xy_max_m=0. These instrumentation checks do not certify physical-metric validity.
- All 15 version-bound source/protocol/configuration files match remote hashes. Local V8 collector and document hashes match their remote/version hashes. Metadata adapter source is V8: 8373f5301e1022d7414c9415305edf08855db2e0a8d3efe000fa0b4dff861109. Protocol hash: 7de595a57b4400069b0bfc238ada5b2deb1f7846574688d0acbe815422743f19.
- Canonical environment-spec SHA256 matches the unchanged ledger: 81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780.
- lineage.json certification is bound_request_at_execution; request exactly matches the recomputed current config/manifest/threshold/version hashes and requests/repeat_a.json. All three output bindings match actual files. confirmatory=false; study_status is amended diagnostic run; thresholds retained.
- Numerical lineage independently recomputed all 12 saved metric records. PASS_ROUNDOFF_ONLY, 0 changed floating values, 0 maximum difference, exact classifications/types/keys retained. Recomputed and saved metric object SHA256 both 20ad574d22b25dd8b0dfc77745f09bd0c0563d998ccf32ba31e3ff4d45578d76. Saved metrics file hash matches certificate and lineage: 0545a27ae568ff4e1d0642e3cf10aff8a16961ca98ea19915e3dfe01adedf3d2.
- Trajectories SHA256 matches receipt and lineage: 5953136a3d482544acf4fc86f85d1c2a2b296f2b6e490457ee19715092ad1b2d.
- Metadata SHA256: 6aae4b5b1770121d0393f4dc5014b1fec86bfba73130f7dc115b3d3ce9781fbb.
- versions.json SHA256 bound at execution: e00bf1d88c3247e79bf4a51165e2ceb644caa5bf698d0e7ea3edaf36d81bb80d.
- Measurement ledger: repeat_a, exit0, 32.495669808238745 seconds. Collector receipt elapsed_seconds=29.68885111808777. No failed witness run or retry occurred.

Verdict: PASS for the exact documented V8 first-batch invocation and required reset/cache/fence/source/numeric instrumentation. No doc-versus-reality divergence observed in this scope. Independent repeat_b remains for root to run. This witness does not certify repeatability, physical-metric validity, method efficacy, task semantics, downstream utility, or real-robot performance. Previous V6/V7 diagnostic failures remain retained and charged by the wrapper's budget ledger.
