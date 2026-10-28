# CURRENT — new100k CPU export sources deployed DISABLED; no actual export

The user endpoint remains VAE100000 + LDM100000. The sole original LDM and
sole endpoint observer continue unchanged. Do not arm another observer, edit
their sources/specs, load a model, or run this export during training.

New sources are `scripts/ldm100k_prefix_completion.py`,
`export_ldm100k_prefix.py`, `supervise_ldm100k_prefix_export.py`, and the new
local activation/launch/collection providers. They implement a distinct100k
provenance path, never the old300k-completion gate. Actual server deployment
and isolated stdlib metadata fixture exit0/reaped/final streams are recorded in
`results/ldm100k_user_handoff_20261028/prefix_export_preparation_v2.json`.
Twenty-six corruption cases were rejected. This did not read production checkpoint
payloads, instantiate a model, create a GPU context or send a production signal.
It is source preparation, not actual tensor/export/quality acceptance.

Current spec `configs/ldm100k-prefix-export-spec.json` has
`export_authorized=false`, `ldm_training=null`. SHA
98a3d4a75423a5ed0bcb497032f53849900e528e023acaaab494caa993dafc0c.
All active original/control hashes were unchanged before and after deployment.
Do not rerun preparation or isolated fixtures merely because a heartbeat fired.

Only after the actual controller reports
`USER_LDM100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED`, its own process,
supervisor and outer have exited0, original workers are gone and common GPU
lock is released, execute once with bundled local Python:

```
provider_tools/activate_ldm100k_prefix_export.py
provider_tools/launch_ldm100k_prefix_export.py --run user_ldm100k_prefix_export_v1
provider_tools/collect_ldm100k_prefix_export.py --run user_ldm100k_prefix_export_v1 --evidence NEW_SHORT_ASCII
```

Activation independently rechecks real closed control/source/exit/checkpoint
evidence, owner1013/no-symlink/ext4/persistent HOME space and16GiB RAM/3GiBdata.
It archives exact disabled bytes before enabling, returns exact activated bytes
with SHA, and does not dispatch. A pending/unknown client or existing dispatch
requires reconciliation, never another launch. Do not call activation before
the real upstream closure, even as a repeated pending preflight.

The900s CPU child has no visible CUDA, CPU0-3, common GPU lock,15s resource
samples and bounded supervision. It loads the selected100k fullstate on CPU,
checks all finite tensors/Adam step counts/clock/closed validation fields and
signed69D/29DOF/data/VAE/numerical binding. It retains saved stage and actual
full-DDPM-history/selected-rollout counters. Snapshot counters can include
post100k memory and must not replace the immutable checkpoint's counters.
Zero counters become findings; they never authorize extra training past100k.

The original `KeyboardInterrupt` and nonzero original process/supervisor/outer
remain real. Actual control closure is checked separately; no claim that the
old300k program completed normally. The export preserves original weights and
full Adam/RNG state, does not restore CUDA RNG or update an optimizer, writes
an inference-only private denoiser with atomic fsync/rename, independently
reloads it and compares a small synthetic CPU forward on both conditioning
branches. Synthetic inputs and exact reload do not establish text semantics.
Require actual `USER_LDM100000_PREFIX_CPU_EXPORT_PASSED` and collector
`closed_execution_verified=true`, CPU process/supervisor/outer0 and lock release.

The128MiB inference upper bound +128MiB log allowance +2GiB HOME reserve and
3GiBdata floor require new actual checks. Existing files/partials/failures are
never overwritten or removed. Future private full-LDM byte backup must use a
new suitable tool and actual capacity, not a VAE-only backup helper.

Independent fresh actual CPU verification, heldout multi-seed text versus
unconditional generation, TMR Rprecision/MMdist/FID, motion FK/physical endpoints,
WBC/online execution and licensed films remain unaccepted. Their real execution
entries and capacity must still be completed, documented and verified. No
staged source or metadata fixture is a substitute. After the genuine endpoint,
continue this work automatically, without asking whether to start or extending
training beyond the user's100k endpoint. All private arrays/text/trajectory/
full Adam/SDK stay server+D; only code and aggregate evidence go to public Git.

Disabled revision2 additionally binds the interrupted original receipt/process/resource SHA, actual sample counts and final after_exit phase. Six new corrupt-resource/exit cases were added; server CPU26-case fixture exited0. Revision1 source/test/spec bytes remain in sources/ldm100k_prefix_export_disabled_v1 locally and receipts/ldm100k_prefix_export_disabled_v1 on the server. Neither revision was activated or used to load a production model.
