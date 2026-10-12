# V45 independent document-following witness

Status: **PASS** — interface and execution binding only.

The exact invocation in docs/original_reference_velocity_v45_20261012.md was executed once. Tool session 26220 returned exit code 0. Launcher PID 2839414 and collector PID 2839437 were observed live before termination. No simulator retry was performed.

## Verified scope and retained cases

- Terminal summary and state are COMPLETE_ORIGINAL_REFERENCE_VELOCITY_DIAGNOSTIC.
- All 72 cases are retained: 24 validation sources, offsets −0.15, 0 and +0.15 m, seed 11. Cases and original source goals match the bound V33 original dataset replay exactly.
- All 38 bound source files and all external motion/semantic inputs match their recorded hashes. The V43 independent witness and original V33 input archive/metadata hashes match.
- Request, versions, execution-time lineage, complete receipt, output hashes and observation trace hashes match. The receipt is bound_request_at_execution rather than a retrospective certificate.
- Generation batch calls and total diffusion calls including warm-up are both zero.
- Reference positions equal the original dataset positions after the documented canonical subtract/add round trip on valid frames. Recorded observation positions equal the applied reference archive exactly.
- Frame clock is exactly 0 through 119. Previous reference positions equal the previous applied frame, with frame 0 using itself. All trace arrays are finite; raw task observations have shape 120 × 72 × 574.
- Native float32 CUDA velocity arithmetic using recorded control dt 0.03333333507180214 reproduces the saved velocities exactly, maximum error 0. A direct NumPy/CPU division attempt gave 4.76837158203125e-7 rounding difference; this diagnostic is retained, not silently treated as exact.
- All reset-to-first-action state differences are zero. The measurement ledger retains the one simulation attempt and charges 49.78750575147569 seconds (receipt elapsed 46.83418297767639 seconds). A separate read-only CUDA saved-array arithmetic check (device cuda:0, tool session 45585, exit 0) is additionally charged a conservative 60 seconds for the whole command, including hash reads and orchestration time. Exact kernel duration is unavailable; visible tool waits total 12.937 seconds plus unmeasured orchestration gaps. Total V45 measurement charge is 109.78750575147569 seconds. The completed batch receipt remains unchanged.

## Calibration counts — no efficacy interpretation

Completed 70/72; condition pass 11/72; physical pass 3/72; tracking pass 58/72; joint pass 1/72. Two incomplete cases remain in the denominator. Zero-offset joint pass is 1/24, and both shifted conditions remain 0/24. Surface-slip proxy passes 3/72, penetration passes 72/72, continuity passes 44/72.

These results do not validate C2 or C3 and do not certify MoCap-derived references as executable ground truth. They do not determine whether failures originate in source motions, tracking capacity, or the proxy metric. There is no semantic quality or downstream training claim.

## Preserved auxiliary issues

A read-only process inventory first failed because remote rg was unavailable; grep then confirmed the live PIDs. An initial artifact read requested a nonexistent execution_receipt.json and produced truncated output; the canonical receipt.json was subsequently inspected. The exact experiment invocation was not repeated. CPU/native arithmetic checks operated on saved arrays only; no simulator or generator was rerun.

Detailed checks and hashes are in DOCUMENT_WITNESS.json. No old results, thresholds, model weights, or cloud Overleaf files were changed.

