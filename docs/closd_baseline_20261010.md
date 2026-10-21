# CLoSD baseline selection and reproduction — 2026-10-10

> **Report publication update — 2026-10-21:** The released-motion evaluation, fresh collection of 20,000 retained episodes, and fresh-motion evaluation are complete. The [complete report and offline video package](../results/closd_baseline_20261010/README.md) now include the 2026-10-17 presentation update. Existing scores are unchanged; no new training was performed. Statements below describing a running batch are historical execution notes.

Baseline: Tevet et al., CLoSD: Closing the Loop between Simulation and Diffusion for Multi-task Character Control, ICLR 2025 Spotlight. Official source: https://github.com/GuyTevet/CLoSD .

Scope: conditional motion generation with physical execution feedback. The simulated subject is the paper's SMPL humanoid, not a G1. This run uses released weights; it does not claim to retrain the full paper. Existing October 1 multitask scores are historical and must not be relabeled as new results.

Selection: CLoSD provides released DiP, untuned/tuned control checkpoints, data, HumanML3D evaluators and runnable recording instructions. Morph is a closer offline physical-refinement reference, but its current public tree/README did not expose the full pretrained pipeline or generator-update reproduction. PhysDiff and RobotMDM official project pages did not provide a comparably complete runnable release in this audit. These are availability observations, not claims that their methods are inferior.

Runtime: nlp03, user ouyanghao, GPU0 RTX A6000. Reuse the read-only Python3.8/Torch1.13.1/IsaacGym Preview4 runtime and closd-r3 overlay from the old humanoid-combat-motion project. Do not install packages or change that runtime. Upstream source is clean and pinned to d8950bdb78312fcd2844f336b3fc079e7368c87f. Assets already staged under HF snapshot de7106b947b6f70700b5320d1cd61fef4a9ebc9b; licenses were confirmed in the previous reproduction. No licensed assets are to be uploaded to GitHub.

Document-following validation (run once, verbatim from PowerShell):

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/closd_baseline_20261010.sh --check'
```

Expected: zero exit, CLOSD_R3_GPU_WITNESS, then CLOSD_NATIVE_CLEANUP_COMPLETE and CLOSD_FRESH_RECORDING_PASS with four saved episodes and a receipt file. This is inference-only (no RL training); a headless simulator is permitted. Do not fix a failed invocation during independent validation. Report its error verbatim. Do not rerun against an existing smoke directory; preserve failed attempts.

After validation, the primary executor launches `bash scripts/closd_baseline_20261010.sh --run` from the remote project in a durable tmux session. Stages: (1) official evaluator on the author's released recorded motions, (2) fresh generation/physical execution with 4096 environments, 20,000 retained episodes, seed10, official t2m defaults, (3) identical official evaluator on the newly recorded motions. Batch32 and ten wo_mm evaluation repetitions remain unchanged. Preserve every stdout, eval.log, resolved config, checkpoint identity and motion hash. Maximum stage runtimes: 2h/4h/2h. Stop on nonzero exit or changed source; do not silently reduce environment count.

Data interpretation: T2M uses a dataset prefix, not text-only generation from an arbitrary neutral pose. The official recording keeps full non-terminated episodes, so saved-motion quality is conditional on survival. Reset/termination counts are recorded separately. They are event counters, not a deduplicated per-prompt success metric. Fine-tuned vs untuned or DiP comparisons require matching datasets, weights, sample counts and evaluation protocol; published numbers are references, never local results.

Storage preflight: system disk 11GiB free, data disk47GiB free. Reuse assets, output only under this project; estimated fresh motions plus evaluation outputs <8GiB. Do not clean other projects. Initial runtime has no package rebuild. Report evaluation reproduction separately from training reproduction, hardware execution and proposed-method utility.

Verified milestones: independent document-following validation passed once (four new physical motions,315 steps,normal cleanup,zero termination events). The ten-repetition official re-evaluation of author-released motions completed: FID1.7533 +/-0.0050, R@3 0.6926 +/-0.0026. These scores are NOT results on newly generated20,000 motions. The fresh full batch is running; consult pipeline.state and receipt.json rather than this static document for completion.

Artifacts: results/closd_baseline_20261010/baseline_report.html and summary.json; raw licensed motions remain on the server. The first three smoke outputs are rendered locally for inspection, without selecting for quality. They are skeleton playbacks of recorded physical trajectories, not native simulator footage. A bounded, single-job finalizer evaluates/renders the completed batch; the local collector transfers only reports, metrics and preview media, and never restarts an experiment.
