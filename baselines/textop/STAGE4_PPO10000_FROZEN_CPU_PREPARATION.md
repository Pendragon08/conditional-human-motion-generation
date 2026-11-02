# Prepared source only — real PPO training remains active

Stage3 remains postponed. The only active model/SDK job is
`user_step4_train_only_ppo10000_train_v4`, targeting10000 PPO iterations and
200000 Adam calls. None of its source/request/budget bytes were changed.
Do not run a CPU model while that job holds the common model lock.

This preparation contains two new, differently named modules:

- `scripts/step4_ppo10000_frozen_protocol.py`: standard-library endpoint
  provenance and fullstate metadata gates. It requires the exact immutable
  formal request, the actual10000/200000 receipt, complete immutable final
  checkpoint SHA, all child/supervisor/outer0 and resource closure, no remaining
  original workers, and no skipped Adam call. An emergency or live progress is
  rejected. The future CPU supervisor must hold the common model lock itself.
- `scripts/step4_ppo10000_cpu_core.py`: future CPU-only fullstate finite,
  model/Adam shape and200000-step audit, native431D actor+learned RMS payload,
  synthetic deterministic mean-action probes and independent actor-only reload.
  No optimizer step, CUDA RNG application or simulator import is intended.

The CPU core is not yet executed. It imports no neural runtime at module
import; its functions may be invoked only after the true endpoint by a future
authorized bounded CPU entry. It compares to the original pinned native
`ActorCritic.act_inference(normalizer(x))`. It preserves the learned RMS mean,
variance, standard deviation, count andeps.01; no new normalization, sampling,
action clamp or critic observation is added. The payload emits raw29D policy
actions before the environment's actuator scaling. Deployment observation
assembly, actuator mapping, heldout tracking and online latency remain separate
real acceptance work. Synthetic CPU equality cannot establish these effects.

The actual installed RSL source bytes were copied without importing them:
actor SHA51ce8c6504a7434f86e5188ccb11e9fc981e46f764c00e32be3e8febe7112ae2,
normalizer SHAf52677617f729c21c7440fd52d69b7e570313836a9deb21519425b806043a311,
runner SHA2f00052c1088f8d13f1b106f25981dc95ff081b2a23841413cca3a7896e0b262.
The runner uses EmpiricalNormalization withuntil1e8; both normalization branches
are audited, only policy RMS enters the inference payload. CUDA byte RNG is
inspected, never restored on CPU. The private fullstate retains critic/Adam/
adaptive sampling/clock/RNG; the small actor payload does not replace it.

`configs/step4-ppo10000-frozen-cpu-preparation-spec.json` is **authorized=false**,
withtraining_binding=null andexport_binding=null. No active training source
imports either new module. The new source contracts parsed as Python3.8 and
passed1 positive and15 negative synthetic metadata-gate examples on local CPU.
Actual checkpoint/model/SDK reads, new optimizer updates, production endpoint
acceptance, export and independent new-process CPU verification are all0.
These checks are preparation, not a trained-policy witness or runtime READY.

The first read-only source copy failed at fullstat equality, which included
read-dependent atime. The original exit1 and streams are retained. A new unique
copy compared device/inode/size/mtime/ctime, then matched the training receipt's
actor/runner SHA. A local command-rendering SyntaxError had0 file/server/model
side effects and is retained. No active source, numerical tolerance, model
weight, resource floor or installed environment was changed.

Remaining implementation before any launch: new CPU entry/supervisor and
exclusive activate/launch/collect bridge; exact final producer binding; actual
available RAM/filesystem budget; immutable private export and all bytes/SHA/
owner/fsync/rename checks; then a real independent fresh-process document pass.
The future budget is CPU0-3, no visible CUDA,16GiB available RAM,900s bounded,
15s resource samples,2GiB reserve+128MiB logs+32MiB inference. It must use DATA,
not nearly full HOME or D. The budget is a declared upper bound, not reserved.

Do not enable this spec by hand, call an old TMR/VAE/LDM exporter, export a
diagnostic PPO state, bypass closure with an emergency, or launch a model while
formal PPO is active. Continue the missing disabled bridge/source work during
healthy training, then execute only after the genuine endpoint. No extra human
confirmation is needed for the already authorized evaluation.

All new private reports/code copies use the accepted C root. D is below2GiB;
no new D writes or cleanup. Weights, trajectories and raw transfer evidence
remain private. Preparation artifacts never substitute for new WBC effects.
