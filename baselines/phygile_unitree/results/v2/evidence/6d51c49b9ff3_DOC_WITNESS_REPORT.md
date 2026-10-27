# Fresh document witness: MuJoCo closed-loop migration v1

PASS for the documented bounded engineering validation. The exact command in MJLAB_CLOSURE_V1_INVOCATION.md was executed once, without repairs, retries, installs, source edits, or server operations.

- Actual process exit: 0.
- UTC interval: 2026-10-22 12:20:01.3923200 to 12:20:27.0054009 (25.6130809 seconds).
- Saved wrapper: run_documented_once.ps1, Windows PowerShell 5.1.26100.9711, UTF-8 WITH BOM.
- The copied environment spec canonically hashes to a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954, matching the provider ledger. Warm reuse only.
- verification.json: COMPLETED_ENGINEERING_VALIDATION, outcome REJECTED_WITHOUT_INVALID_COMMIT; actual_mujoco_warp_physics=true; actual_diffusion_sampling=true; optimizer_updates=0.
- Both actual diffusion proposal archives contain raw_features_262 with shape [52, 262] (two network context frames plus 50 newly proposed frames). Both proposals were attempted against actual physics, terminated after one recorded frame and were rejected: incomplete_continuation, failure_termination, mpjpe_above_tolerance. Their MPJPE values were 0.3926215555626271 m and 0.39316207382514584 m. Zero generated frames were committed; committed_prefix has one initial state only.
- The separate held-initialization feedback proxy accepted two one-frame windows. Their MPJPE values were 0.00826349488046638 m and 0.018353520141933036 m. These are proxy references, not learned generation or dataset truth.
- Fresh-scene seeded replay over ten frames had max absolute native-state delta 0 and body-position delta 0 m. Accepted-history replay also had max state difference 0. Seven simulator scenes were fresh.
- All 29 immutable snapshot source hashes matched before and after. All 255 executed source manifest entries still independently match disk; its manifest SHA-256 is 7a68de6eac8ff105bffa25cf996cbea2b9430e2ea3e4139e1d6656cd65100433.
- The frozen policy checkpoint remains bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336; its actual file hash and normalization hash match verification. Generator parameters are reported unchanged. Every saved raw array hash was independently checked.

No material document-versus-reality divergence was observed. Nonfatal stderr contains the upstream torch.jit.script deprecation warning. PowerShell 5.1 labels redirected native stderr as NativeCommandError; the separately captured actual Python exit code is 0. stdout.txt and stderr.txt are the complete PowerShell-captured streams (UTF-16LE), not excerpts.

The result supports untrained, unconditional 262D diffusion wiring with real simulation feedback, rejection and replay. It does not support stable or semantic motion generation. Licensed arrays remain only in the private run output directory. This witness directory contains hashes, metadata, metrics and logs; no raw motion arrays or checkpoint copies.

Evidence: command.txt, run_documented_once.ps1, launch.json, actual_exit_code.txt, process_result.json, stdout.txt, stderr.txt, verification.json, independent_artifact_verification.json, source_hashes_before.json, source_hashes_after.json, source_integrity.json, source_manifest.json, executed_sources.json and copied invocation/environment documents.