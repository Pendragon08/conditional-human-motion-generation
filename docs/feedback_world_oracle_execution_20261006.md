# W-only physics-oracle upper bound — exact execution

This is frozen-policy MuJoCo inference using the unchanged env ledger
`.aris/compute/ssh-mujoco.md` and spec hash `91455383`. No installation, PPO,
extra robot IK or baseline rerun. It follows predictive H10 PASS plus complete
coverage, but does not yet execute a learned world-model controller.

Fresh independent doc-follow witness, verbatim from Windows PowerShell:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_feedback_world_oracle_20261006.sh witness"
```

Must exit0 and print `ORACLE_WITNESS_PASS` for Kick and Punch. This uses only
validation ref36, executes a real force prefix, checks full MuJoCo integration
state cloning, candidate rollouts leave the actual state and previous action
untouched, and finite bounded candidate costs. No test metrics or model fitting.
After PASS, run the same wrapper with `eval` once. Do not edit a running script.

Fixed config `configs/feedback_world_oracle_20261006.json`: heldout refs46–65 of
both tasks; prefix shares actual frozen PPO plus world-y80N force at steps50–59.
All arms start from the identical complete state at step60 with future force0.
Compare original PPO, clipped0.8/0.2 smoothing, and physics-oracle candidate
selection (candidate seeds42/43/44, not independent PPO training seeds).
Candidate set is original action plus two +/- mirrored Gaussian offsets,
std0.06/clip0.12 in27 physical actions; final2 outputs unchanged. At10Hz choose
using H10 (0.2s) oracle rollouts and apply offset for5 real steps. Frozen PPO is
recomputed each predicted/real step. Parameters/costs are fixed before test;
ref36 witness is interface validation, not validation optimization.

Intervention runs steps60–149; afterwards every arm returns to the frozen PPO,
with original reference until step500 or fall. Primary recovery joint RMSE AUC
has a fixed2rad penalty for unobserved recovery steps after a fall. Report
virtual target success in the original +/-0.1s event window, fall rate and
action variation. A virtual goal is not entity contact. Gate: >=5% paired
reference-macro AUC improvement, bootstrap lower95%CI>0, virtual success drops
<=5 percentage points, and no increase in fall rate. If FAIL, stop learned-W
control development for this interface. Oracle PASS is an upper bound only;
it cannot count as a learned-world-model or full innovation result.

Expected outputs: `results/feedback_world_model_20261006/oracle_witness.json`,
`oracle_records_partial.json`, `oracle_summary.json`. Once outputs exist,
no duplicate eval or overwrite. Healthy running inference is not restarted.
