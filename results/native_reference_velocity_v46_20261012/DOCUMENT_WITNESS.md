# V46 independent document-following witness

**Overall document contract: FAIL.** The exact documented command ran once, without retries or source edits, and exited 1. Both 72-case measurement batches completed successfully before the launcher failed at line 60 with `KeyError: generation_noise_sha256`: it attempted to read generation noise from prior replay metadata. Launcher process 2842814 is absent. `state.json` remains RUNNING and the expected terminal `summary.json` is absent; neither is treated as evidence of a live job.

## Distinct batch/interface validity

- Both requests were independently reconstructed; strict execution-time lineage, receipts, source bindings, prior witnesses and external motion/semantic input hashes match. Original targets, headings, texts, 72 ordered cases and validation source grouping are unchanged. Native construction uses the same generation-noise hash as V43 planned construction. Replay has no generation noise field and none is invented.
- 120 frames × 72 inputs are retained, including failures. All arrays are finite, time indices 0..119 match, previous positions equal the prior applied reference and frame 0 velocity is zero. Native float32 velocity is reconstructed exactly with float32 subtraction and rounded float32 reciprocal multiplication (s_dt cast to float32; reciprocal 29.999998092651367). An initial CPU comparison incorrectly used 30.0 and is explicitly retained as an arithmetic characterization in JSON; no simulation was repeated.
- Native construction and fixed-reference replay have bit-identical trajectory, observation-trace and initial-state arrays at all 120 frames, including all 72 valid inputs per frame. This is a paired interface observation for this validation seed only, not general efficacy or causal evidence.
- Fixed replay reference equals the canonical transformation of recorded native positions exactly. All 72 metrics were independently recomputed under the execution NumPy 1.23.5 overlay and match saved values exactly. An initial bare NumPy 1.24.4 recomputation had a small surface-proxy numeric difference and is retained in JSON; the documented runtime was used for the valid comparison, without altering scores or tolerances.
- The separate V47 CPU recovery summary was independently checked against inputs, source hash, output hashes and counts. It uses no new generation or simulation calls. It does not turn the original failed documented invocation into a successful run.

## Descriptive measurements, not efficacy

| Batch | Completed | Condition | Physical | Tracking | Joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native construction | 72 | 10 | 0 | 51 | 0 |
| Native fixed-reference replay | 72 | 10 | 0 | 51 | 0 |
| V43 planned fixed-reference replay | 69 | 27 | 0 | 54 | 0 |

The successful batches were charged 49.340134812518954 and 48.89001667127013 measurement seconds. No additional GPU jobs, source edits, installs, retries or Overleaf changes were made. No candidate mechanism, G1 or downstream training gate was bypassed. No innovation is confirmed.

Machine-readable checks, exact hashes, lineage, all-field comparisons, initial arithmetic/runtime audit failures and execution handles are recorded in `DOCUMENT_WITNESS.json`.
