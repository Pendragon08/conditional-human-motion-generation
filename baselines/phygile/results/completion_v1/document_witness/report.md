# Independent completion document witness

The documented invocation passed once on the original authorized server. All seven stages completed; SSH returned 0, stderr was empty, and stdout contained exactly one full line `PHYGILE_COMPLETION_VALIDATION_COMPLETE`. This is an engineering-interface validation, not a claim that the paper's main experiments are reproduced.

The earlier hold occurred before invocation. The initial preflight remains preserved; the new preflight at 2026-10-17 14:29:00 UTC confirmed that `document_run` did not exist and GPU 0 used 2968/49140 MiB, as permitted by the document. No existing jobs were stopped.

Executed verbatim:

```sh
cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017
python3 scripts/validate_completion.py --name document_run
```

The pipeline ran from 14:29:27 to 14:33:18 UTC on 2026-10-17 (231.14 seconds). The witness did not inspect implementation code, modify code/docs/environment/thresholds, rename the run, or rerun the experiment. The environment spec is `fa5883c8`; the actual main checkpoint hash matched `bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336`.

| Stage | Verified actual result |
|---|---|
| prepare | COMPLETED, exit 0; stage receipt and log present |
| regression | PASS, 13 mathematical/counterexample groups |
| codec | 24 sequences (20 train, 4 test); normalization fitted only on 20 train sequences; 24 feature NPZs and metadata files present |
| curriculum | Valid evaluation of 8 first episodes per clip: train 0/8, test 3/8, total 3/16 full clips (18.75%) |
| expert_mask | Requested and reported `active_experts=2`; 2 recorded episodes, 0 successes; compatibility only |
| training_hook | 32 environments, four PPO iterations; independent evaluation at iteration 3; 0/2 successes; `promote=false`, `minimum_file_success_below_threshold`; level stayed 1 |
| closed_loop | Real 146,169,702-parameter diffusion sampling and Isaac execution; two generated 50-frame candidates rejected; zero committed new frames |

Codec maxima were 6.5937806e-09 m root-position error, 4.1274573e-08 rad joint error, and 9.6294252e-07 m FK consistency error. The output explicitly labels this `self_supervised_proxy` and `author_exact_codec_reproduced=false`.

Curriculum MPJPE was 0.266367927 m and MPJAE 0.235589333 rad, equally weighted per requested clip and censored at termination. Independent raw-array recomputation confirmed 209 train observations across 8 failed episodes and 862 test observations across 8 episodes, of which 3 reached all 149 frames without failure. The expert-mask check contained 30 observations across its 2 failed episodes. Episode lengths and success counts matched the reports; these three raw archives contained no nonfinite numeric values. The largest absolute difference between independently recomputed and reported errors was 2.4742e-08.

Each generated candidate recorded only 1/50 expected frames, with MPJPE 0.426518343 m and 0.428358649 m. Both decisions recorded incomplete continuation, failure termination, and error above tolerance. The committed prefix remained identical to the initial prefix. Separately, the artificial held-initialization proxy accepted two one-frame windows (two control frames total), with MPJPE 0.022785566 m and 0.043553914 m. The ten-frame repeated-rollout witness and one accepted-history replay both recorded zero state difference. The method was fresh-process accepted-history replay, not private PhysX state serialization.

Primary integrity checks covered 82 expected nonempty files, all 151 result-file hashes (5,483,866 bytes), and all 69 NPZ archive CRCs and NPY headers. Reported raw-data, checkpoint, and normalization hashes matched. All seven simulator worker requests/results and the external training decision/evaluation primary files matched. The preserved audit is `artifact_inspection.json`; independent raw computations are in `raw_primary_checks.json`.

The first local-copy check found two absent fixture NPZs that were valid remote symlinks. The parent materialized their targets as local regular files without modifying the remote evidence. An independent repeat then matched all 151/151 local files to the remote manifest. The original discrepancy remains in `local_copy_verification.json`; its resolution is in `local_copy_verification_resolved.json` and `../fixture_materialization.json`. No invocation/document divergence remains.

Limits: the generator is random-weight and zero/unconditional-text, with a chosen eight-step schedule, URDF projection, and engineering bounds. All generated windows were rejected, so this run demonstrates no accepted generated continuation or generated terminal-boundary acceptance. The separate accepted two-frame proxy is not a one-second generation result. The expert-mask check does not measure the untrained second expert's policy quality. Curriculum IDs are arbitrary fixture IDs, not difficulty annotations; 80 Adam steps are recorded as expected, not independently counted actual steps. No paired-data generator training, semantic text-to-motion quality, full Stage I/II GMT, closed-loop PPO adaptation, paper-scale results, or hardware deployment is established. Memory numbers refer only to the PyTorch allocator. Subsequent supplements are outside this witness and its archived execution source.

Transport and provenance are preserved in `invocation.json`, `invocation.stdout.txt`, `invocation.stderr.txt`, `invocation.transport_exit.txt`, the stage receipts, and the archived execution source identified by `report.json`.

