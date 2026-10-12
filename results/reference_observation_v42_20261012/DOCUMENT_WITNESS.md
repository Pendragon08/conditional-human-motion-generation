# V42 fresh document-following witness

PASS for the documented validation-only passive measurement interface. The invocation ran exactly once, exited 0, printed COMPLETE_REFERENCE_OBSERVATION_DIAGNOSTIC and reached COMPLETE_INTERFACE_DIAGNOSTIC. This does not validate innovation efficacy, C2/C3, normalized policy observations, or a causal effect. Prior C1 remains negative.

## Witnessed command

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reference_observation_v42_20261012.py --run
```

Local exec session 93582; remote launcher PID2833056/PPID2833055 started Mon Oct12 00:15:14 2026 (server local). First collector PID2833079 and native child PID2833109 were observed. The same session was followed until exit0; terminal process inspection confirmed launcher absent. No retries, threshold/source changes, rebuilds or other-job interventions.

Preflight memory: GPU0 699/49140MiB; GPU1 8842/49140MiB. Launcher recorded GPU0 utilization0%, consistent with documented idle viewer policy. Both logs ended PHYSICS_GUIDED_NATIVE_CLEANUP and PHYSICS_GUIDED_COLLECTION_PASS 72 0; final progress was 120 72 44.74 (planned) and 120 72 45.14 (replay).

## Independent verification

All 34 version-bound sources and four historical V28 anchors hash-match. Local and remote document hash 967d7e02edd6d9343d1ba17feb72418344bcf402db858dfbf265c39e948442f5. Existing warm-runtime canonical spec hash 81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780 matches the existing .aris/compute/closd-baseline-20261010.md ledger. AGENTS.md and initially assumed .aris/compute/nlp03.md were absent; no new environment setup was needed.

Each batch contains the exact ordered 24 validation sources × three frozen offsets [-0.15,0,0.15] =72 inputs, seed11; metadata/metric count72; all initial attempts retained. Both bound_request_at_execution lineages pass guard_batch, request/output hash checks and completed receipt checks. Trace hashes match metadata.

Reference/previous/velocity trace shape [120,72,24,3]; raw task observation [120,72,574]. Every field finite. Reference clock exactly0..119, trace reference equals each saved trajectory reference. raw_reference exactly equals planned trajectory reference. Paired references agree on every common-valid frame; replay canonical reference error0. Original goals agree.

Planned retained two terminations with lengths95/109 (70others120); replay retained one length118 (71others120). No input was discarded.

Ledger: planned49.492541167885065seconds and replay49.89222070388496seconds, both exit0. Cumulative measurement3878.5646072737873<14400seconds; total14657.29460927844<259200seconds. All recorded outputs7140245471<16106127360bytes; free /data38463463424>21474836480bytes. V42 output88067407bytes.

## Passive recording and interpretation

New planned trajectories.npz exactly matches historical V28 val_planned_s11_r0 byte hash 45486979e705ce28f3e5e10976814ce901f3008ea902e5e23d326a5c0a4311bd. All nine arrays are bit-identical; cases and explicit generation-noise hash also agree. This supports passive instrumentation in this witnessed configuration.

At frame60, positions agree but previous pose maximum difference0.074100062251091m and returned reference velocity difference2.223001718521118m/s. Raw task observation already differs at frame1 by0.043733835220336914. Field differences cannot establish their causal behavioral effect or innovation efficacy. The recorder captures raw task obs before native pre_physics_step, not internally normalized policy observation.

## Output hashes

| File | SHA256 |
|---|---|
| summary.json | 063de9fd2d0bc08cff822dff8e0ee7c95d5ed9c0ef2be059cc33f427f5e7594f |
| state.json | b956dffffc157080ae30fb870ab96cf191c07d530b090bff37ecee413ebbbe67 |
| versions.json | 00ae3501899a92d524509f7661d91daf23c505434e69ab37c8b2817050e5e54b |
| budget_ledger.json | fd7a490fa968a402c455f1e7e1f1f3b9ced5528ad23320fa5ea2420f754ee8b5 |
| planned observation_trace.npz | 17b23e41c7c973bab5583c3d8fc073d7e44df5e644ae5a3160f6ae3d5167c638 |
| replay observation_trace.npz | 9033536f0a342935cf5e9bc56fe0431fed016c9121cd741067d9acc5a184578c |
| planned lineage.json | 197e2f4f45b31c68ae108847175ce0b6125e488415b8354cd88abc4f67a0b1dc |
| replay lineage.json | 0bafa38714787b74eab3233071330b4cf126da02ad5e1b444db78c8faba1f666 |

DOCUMENT_WITNESS.json contains the full witnessed stdout summary, all source/output bindings, verification shapes, retention, budget rows and scope flags. Licensed arrays remain on the server.

## Divergences and inspection errors

No documented-command or expected-terminal-state divergence observed. Auxiliary read-only inspection harnesses encountered ReferenceError: btoa is not defined and bash syntax error near unexpected token base64.b64decode. Successful stdin-only read inspection followed; GPU invocation was never retried. The first local report-save attempt exceeded the Windows command-line length limit and wrote no file; the two authorized reports were then saved through the file editing tool. No failed GPU attempt was repaired.

