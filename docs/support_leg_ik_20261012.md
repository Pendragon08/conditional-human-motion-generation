# Bounded leg-coordinate diagnosis before remaining candidate efficacy

This is an engineering intervention, **not a fourth innovation candidate**.
The source is the fixed V51 seed11 shared first construction. The cached V63
baseline is reused byte-for-byte, including its original cost and retained
initial FAIL plus separate numerical-contract addendum. No historical output
or score is overwritten. Internal validation comprises24 sources×3 offsets;
it includes upstream official train/test and is not an untouched test set.

Keep the first61 poses, last pose, original goals, pelvis/wrist task joints,
all upper-body coordinates and generator/controller weights unchanged. For each
leg separately, use the prior frozen simulated support/velocity proxies to
solve the same bounded horizontal translation objective. The ankle and toe
move together by at most10cm. A classical analytic two-link IK reconstructs
the knee while preserving the original per-frame thigh/calf lengths. Reject
unreachable or knee-shift-above10cm requests via fixed dyadic backoff. No
contact-point truth, joint-angle limit or full-dynamics guarantee is claimed.

Arms: cached baseline; Cartesian foot translation with original knee; the same
accepted foot translation plus two-link knee reconstruction. Cartesian control
is a positional-representation ablation which can violate segment lengths,
not a proposed robot-feasible generator. All72 inputs, including incomplete
construction sources, are retained; incomplete sources receive no repair.
Endpoint reference equality is not actual endpoint success. Root and heading
conditions are never redefined. Classic IK is not a novelty claim.

Warm runtime, NumPy1.23.5 overlay, pinned CLoSD/DiP and tracker as parentV63.
No installs, new training, VLM API, G1/robot IK, or Overleaf experiment writes.
New simulator calls only two, at most300seconds each and600seconds total;
stage/global budgets reserve the full timeout including the separately charged
numerical CUDA adjudication. Disk≥20GiB, round outputs<15GiB. Every failure and
charge is retained. Budget ledger is not interpreted as algorithm efficacy.

Fresh independent document follower executes this **once** after preparation:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_support_leg_ik_20261012.py --check
```

Expected exit0 and `SUPPORT_LEG_IK_DOCUMENT_PREFLIGHT_PASS`. This is CPU-only,
not a simulator execution. Independently inspect every repaired reference,
baseline/request lineage, source/code/model hashes, paired ankle/toe shifts,
unchanged prefix/endpoints/task joints, ≤10cm shifts, ≤1e−5m two-link length
error and finite values. Save DOCUMENT_WITNESS.json/.md under this directory
with status, preflight_sha256, source_sha256 and document_sha256; any divergence
is FAIL and must be retained without improvising or rerunning.

After that bound PASS, root runs the same invocation with `--run`, once. It
must complete both arms with all72 metric records and explicit receipts. For
analysis recompute fixed thresholds per raw row, report finite denominators,
paired source-level deltas, exact exported reset-state comparisons and costs.
If no physical/joint gain, stop this intervention; if a component improves,
report only a one-seed diagnostic and no confirmed innovation/downstream gain.
C2/C3 original launch gates are not bypassed by this diagnosis.
