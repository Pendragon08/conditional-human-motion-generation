# Research findings

## 2026-10-06 — Execution feedback and learned dynamics pilot

- CLoSD executed-motion prefix feedback can guide a DreamControl extension, but the 263D HumanML bridge is not interchangeable with G1 states. Feedback to the generator is distinct from existing PPO proprioception.
- Sparse residual feedback changes the frozen OmniControl output and fits the supplied nominal context. Three Kick references/one diffusion seed do not establish control benefit. DDPM1000 batch latency94–107s does not meet a2s replanning budget. Future-goal error is not monotonically improved by correct versus shuffled feedback.
- A learned action-conditioned G1 predictor passes the locked H10 forecast gate on reference-heldout real MuJoCo data:15.73% improvement over best simple predictor,20.72% over state-only. This is recorded-action prediction, not learned-MPC success.
- Five-candidate fixed-cost simulator oracle completed200trials, formalPASS:29.66% recoveryAUC gainvsPPO; post-hocvsSimple17.93% withreferenceCI>0. Kick isworsethanSimple onAUC; overallnotallmetricsbetter. This is physics-oracle, notlearnedW.
- Learnedmulti5/single control completed240unique realMuJoCo trials, normalexit0. Primarymulti5 gateFAIL:15.21% AUC pointgainvsPPO CI[-4.72,28.33]%,1.07%vsSimple CI[-16.12,13.87]%, fall42.5%vsPPO40%. Kick deteriorated whilePunch contributed aggregate pointgain. Scoring72.32msmean/250msmax and1.17%over100ms, excludes otherI/O. Do not promotepredictionPASS tocontrolPASS, switchprimaryarm, retune exposedtests orstartF×W/formalPPO. Fullinnovationremainsunvalidated.
- Absolute sparse bridge failure is preserved; after-origin/delta exploration on exposed test references is explicitly exploratory.
- Main fitter computed all planned results, but launcher exited2 after a running wrapper was edited. Post-start kernel witness is not pre-launch verification. No retraining or silent result/status rewriting.

Detailed numbers and evidence: [feasibility report](reports/21_feedback_world_model_feasibility_20261006.md).
