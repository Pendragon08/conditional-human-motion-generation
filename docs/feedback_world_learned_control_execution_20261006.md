# Learned-W frozen-policy control — exact execution

Reuse unchanged MuJoCo ledger/spec91455383; no install, PPO, extraIK or existing
baseline rerun. Oracle formalgate PASS and independent oracle-vs-simple
diagnostic CI positive permit this staged pilot. Predictive, oracle and
learned-control tests reuse heldout refs46–65: no final independent confirmation.

Fresh independent witness must run exactly once:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_feedback_world_learned_control_20261006.sh witness"
```

Must exit0 and print `LEARNED_CONTROL_WITNESS_PASS` for three model variants and
both tasks on validationref36. Fittedcheckpoint hashes verified, finite H10
costs, physical candidate clip.12, actual fullstate andpreviousaction unchanged.
No test trial or model fitting in this witness. Afterwards run wrapper `eval`
once, preserving partial failures. No active source/config/script edits.

Candidate set, root/joint/up/offset cost,4*remainingstep shortfall penalty,
prefix/push/init/reference, H10,5stepdecision period,60–149intervention and500
step episode are identical to the fixed simulator-oracle protocol. Learned
predictions replace only forward physics scoring. At each imagined step the
same frozen PPO recomputes its action using predicted state and fixed future
reference. No ground-truth future state/force is given to the learned model.
Models multi5/single eachseeds42/43/44 are compared,240new episodes.
The state-only model cannot distinguish candidate dynamics; positive residual
cost makes its controller identical to the frozen PPO. The witness verifies
zero selection; analytical reuse of PPO replaces a redundant physical rerun.
Previously completed PPO/simple/oracle200trials are reused, neverrerun.

Primary multi5 gate: recoveryAUC>=10% betterthanPPO with pairedreferenceCI>0,
alsoCI>0 vs simple; virtual targetsuccess drops<=5points, fallsnotincrease.
Timing records mean/max and0.1sdeadline overruns. This is synchronous offline
physics evaluation; cannotclaim real-time control if deadline violated.
Auxiliary removal models are samecandidatebudget; no parameter retuning on
test. All source/config hashes fixed at launch. Expected outputs under
`results/feedback_world_model_20261006/learned_control_*.json`.

Even PASS is a boundedW-only pilot, notF×W, physicalcontact, newreference/task
generalization, onlineF, sim-to-real or fullinnovationvalidation.
