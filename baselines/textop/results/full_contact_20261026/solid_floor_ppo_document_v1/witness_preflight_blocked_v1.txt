# Independent document witness: preflight blocked before launch

The documented `solid_floor_ppo_document_v1` invocation was **not executed**. No GPU job, training iteration, optimizer update, checkpoint or physical filtering was started by this witness. The failure is preserved and must not be described as a completed document run.

The witness read `SOLID_FLOOR_PPO_ENVIRONMENT.md`, its declared specification, the warm environment ledger, supervisor/probe, collector and named dependencies. A bounded read using `provider_tools/session.py` independently established:

- Author supervisor receipt exists: `child_exit_code=0`, `timed_out=false`, `ppo_validation_completed=true`, elapsed 283.853578925133 seconds.
- The document run directory and its actual exit receipt were absent.
- There were no active personal SDK workers among the supervisor's blocked worker set. Paused processes were preserved.
- GPU 1 had 40,726 MiB free out of 49,140 MiB. Host RAM and personal data-disk free space exceeded declared minimums.
- All 8,278 original reference bindings were reread and verified, with aggregate order fingerprint `0b786ba08bf8f33aa92cb4e91ed72cd5b56159c4e251cedf88f429269141c1f7`.
- Both named private input hashes matched. The warm environment canonical hash was `b7a3bf0c`.

The document requires checking the specification SHA and listed source SHAs before invocation. Two declared source hashes fail against the identical local/server file bytes:

| File | Declared SHA256 | Actual local and server SHA256 |
|---|---|---|
| solid_floor_ppo_probe.py | cb7baf7d33ab8b6eae51ebc8ee5261a79719bf6f425f11fa18bc51a810e273d4 | 3129f2f071f9e333d3548d61b2278f9c650425a21402054e070ab321e7d6f28b |
| supervise_solid_floor_ppo.py | ecdb1c4fee84aa330c73aec86aa29b2b1a02b74fb57653438a5a05ddeb3fe7e5 | b59bcab493a7207953dc4e9208d8217356c409e0f7aa6786139a4b2197d9d417 |

Both declared hashes equal the corresponding local file's LF-normalized byte hash; deployed files contain CRLF. This diagnoses the mismatch without treating normalized hashes as the documented exact byte checks. The specification's actual local/server SHA is `75a08f04ebb91331eb2888a3cc3a51ac422c374680f69796ddae9b5929326186`.

An additional dependency mismatch exists for `configs/paper-step2.json`: local SHA `d668360a8cbc1d4ad48845467841f3d4d9804f291a86790201ac3de74fec42dd`; server SHA `e0466458fac00280d4ddb413622c4c71d9f62f4f07084026757b405ef7ddc1f4`. Local LF normalization gives `45fef375dcc0103d8cd747aff3893908475ed6179afc3282b7ae6475509a4f39`, so this is not solely line endings. Its semantic difference was not independently read at this stage.

The other six specification-declared source hashes passed. Of the 18 compared dependency files, only `paper-step2.json` differed between local and server bytes. `preflight.json` holds exact aggregate evidence and resource measurements. The local preflight process exited 1 at its final source-hash assertion after saving that evidence.

Actual document-run iteration/update/physics/reward counts, loss/trace finiteness, initial-state/action equality and native-close/fallback behavior are **untested** because the required prelaunch hash checks failed. No document-run private data were produced and the collector was not called. No dependencies, production configurations or main reports were modified, and no existing run was retried or resumed.

This is a ten-iteration diagnostic witness only. Formal 40,000-iteration training, corpus stability acceptance, physical filtering and generator training remain unvalidated.
