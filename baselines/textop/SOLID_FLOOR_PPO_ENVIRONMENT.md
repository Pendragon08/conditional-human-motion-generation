# Solid-floor candidate: actual PPO validation, formal gate still disabled

The warm tracker environment b7a3bf0c is reused without installing or modifying
shared packages. The original production configuration remains unchanged. This
candidate replaces the locally authored infinite Plane collider with a static
4000x4000x0.2m box whose top is z0; friction1, restitution0, multiply modes and
robot self-collision remain enabled. The original8278 references, initial state,
seed, reward, termination and paper PPO settings remain unchanged.

Declared profile: configs/solid-floor-ppo-spec.json. Before each invocation verify
its SHA and the listed source SHAs match the server. The supervisor rejects an
existing run and any active personal SDK worker, holds the common flock, checks
GPU1/RAM/disk, restricts CPU0-3 and bounds execution1800s. Torch cap0.65.

Author invocation (execute once):

```
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_solid_floor_ppo.py --run solid_floor_ppo_author_v1 --case solid_floor"
```

Fresh independent document invocation, only after author actually exits:

```
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_solid_floor_ppo.py --run solid_floor_ppo_document_v2 --case solid_floor"
```

Expected: actual childexit0, ten collection/PPO iterations,200 optimizer updates,
1966080 transitions,960 physical samples and240 reward samples, nonzero parameter
change, ten finite loss rows. No checkpoint is written. Inspect all nonfinite
counts and retained diagnostic alerts as well as initial-state/action differences;
completion alone is not stability or motion acceptance. Diagnostic thresholds
select snapshots only, never drop data or clip reward. Preserve failures and
SDK native-close/fallback distinction. Full long-run convergence is not tested.
The paper40k filter, real two-pass physical evaluation and generator remain gated.

Collect only after actual exit, from local workspace:

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/collect_contact_diagnostics.py --run solid_floor_ppo_author_v1 --family full_contact
```

Use the document run name for its collector. The collector stores only aggregate
receipts/trace publicly and SHA-verifies private logs/snapshots on the user D drive.
A fresh witness should compare declared source hashes, actual counters, initial
matching, aggregate trace, and document-vs-reality; do not edit results or scripts.


2026-10-26 preflight correction: document_v1 dispatched no GPU job. Its SHA check
found a LF-string/CRLF-file mismatch in the two declared source hashes. Corrected
spec hashes now identify actual local/server file bytes; neither executable nor
numeric settings changed. Original author launch spec and divergence evidence
remain in results. The paper-step2 discrepancy was only user-instruction/storage
metadata; data_scope and numerical settings matched, and exact local bytes have
now been synchronized with the old server version preserved. Use fresh v2 above,
not v1. This correction itself is not an independent execution pass.
