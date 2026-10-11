# V28: complete C1 paired comparison after measurement repair

Prerequisites: V26 full validation PASS, V27 fixed-reference replay complete, V24 calibrated validation not a total physical-quality floor. Preserve all earlier failed versions; no tolerance/threshold changes, no Overleaf result writes.

This is the original C1 algorithm and full comparison, not a new innovation. Frozen DiP and tracker, all 48 preselected test sources, three offsets and three seeds: 432 replay inputs per arm. Ten arms: raw, ordinary, random_best, native_single, native_best, proposed, no_feedback, shuffled, planned_prefix, tracking_only. Maximum three candidates; original text, target, time and duration retained. Equal-budget controls and four ablations remain intact. Original source-cluster bootstrap, semantic/diversity guard and progression gate remain unchanged. Historical exposure/retrospective measurement repairs mean this is not a pristine confirmatory study.

To satisfy the 15 GiB cumulative output bound, the collector retains native COM properties, surface-proxy metrics, trajectories, sensors/actions and all failures, but omits dense diagnostic contact manifolds already calibrated in V16/V20/V24. An empty diagnostic contact JSON is explicitly tagged as recording disabled, never as no physical contacts. Before efficacy runs, exact trajectory/sensor/action/metric equality against V24 planned seed11 must pass.

## Exact fresh document check (one invocation only)

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_condition_feedback_formal_v28_20261012.py --check
```

Expected exit0, `CONDITION_FEEDBACK_FORMAL_DOCUMENT_CHECK_PASS`, all trajectory/sensor/action fields bit-identical to the frozen V24 batch and metrics identical. Check native body COM recorded, dense recording flag false, original-goal hash unchanged, source/doc/config/weights bindings, calibrated metric verifier and one 72-input measurement ledger entry. Save independent witness under `results/condition_feedback_formal_v28_20261012/DOCUMENT_WITNESS.md`. Never retry or fix in witness role. Passing check is not C1 efficacy.

After independent positive witness, root may run:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_condition_feedback_formal_v28_20261012.py --run
```

Run is resumable only through bound complete receipts, never incomplete-output retry. Prior charged time and output size are carried forward, GPU0 is used only when spare capacity is available, data free>=20GiB and outputs<=15GiB, total<=72GPUh and stage caps unchanged. No generator/tracker training, paid API, hardware or automatic G1/downstream execution. Final status is COMPLETE_GATE_NOT_MET or HUMAN_GATE_PASSED_PENDING_VALIDATED_G1_AND_TRAINING_ADAPTERS. Only the latter permits next-stage validation, and neither is a confirmed paper innovation without substantive literature difference and downstream evidence.
