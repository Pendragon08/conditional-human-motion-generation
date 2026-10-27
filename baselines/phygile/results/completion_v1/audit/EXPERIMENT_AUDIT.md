# Experiment integrity audit — PhyGile completion_v1

**Overall verdict: PASS (provisional acceptance).** All round-1 integrity findings are resolved within the authorized follow-up scope. No blocking defect or unresolved evidence warning remains. Acceptance stays provisional because the review is same-family; a different-family review is still recommended before final acceptance.

Round 1 is preserved at results/completion_v1/audit/round1/. This follow-up read only the five authorized revised artifacts: the report, its builder, reimplementation/README.md, static_release_manifest.json, and static_release_SHA256SUMS.txt. It ran no experiment and did not reopen the broader evidence set. input_manifest_final.json contains 57 entries. Its only differences from the 54-entry round-1 manifest are the two revised round-1 files and three newly included files. All five current byte counts and SHA-256 values match the final manifest.

## A — Reference and target provenance: PASS

The revised report preserves the explicit limits: full PhyGile and trained text-to-motion remain unreproduced at reports/PhyGile课程评估与262D闭环补全报告.md:1-13; the terminal target remains nominal and not author-verified at lines 46-48; the held-pose test remains self_supervised_proxy and is not semantic generation at lines 50-56. The new data-validation wording correctly states that metadata checks cannot prove training-data authenticity and still require provenance audit at lines 58-62. reimplementation/README.md:3-17 preserves the same model/proxy and author-equivalence limits. No model-derived target is promoted to independent measured truth.

## B — Normalization, denominators, and success: PASS

The revised report retains file-equal aggregation, termination-censored raw metre/radian errors, full-clip success, and train-only normalization at reports/PhyGile课程评估与262D闭环补全报告.md:15-24, 32-42. It now states that MPJVE is stored per file but excluded from aggregation and promotion at line 19. The builder emits the same statement at scripts/build_completion_report.py:48-50. No denominator, threshold, success definition, or metric implementation changed.

## C — Claims, values, statuses, and identities: PASS

Round-1 warning C1 is resolved. results/completion_v1/static_release_manifest.json:1-26 lists four frozen assets with byte size and SHA-256. results/completion_v1/static_release_SHA256SUMS.txt:1-4 contains the identical filename/hash pairs. Independent static hashing matched every declaration, including raw results at 9,533,668 bytes and SHA-256 1c52bd6403bd72b5399395754977e0296306124b239c0dceabc450d00b82da13.

The revised report and builder differ from their preserved round-1 bytes only in the scoped documentation changes: MPJVE labeling, the provenance caveat, and the provisional audit note. Their final hashes and those of the three newly included files match results/completion_v1/audit/input_manifest_final.json:136-139, 196-199, 220-223, 334-344.

## D — Evaluation control flow and diagnostic metrics: PASS

Round-1 warning D1 is resolved through the permitted explicit-label path. MPJVE remains a per-file diagnostic, and the public report now forbids claiming that promotion includes velocity error at reports/PhyGile课程评估与262D闭环补全报告.md:17-20. scripts/build_completion_report.py:48-50 makes the limitation reproducible. No evaluator/control-flow code, active mask, state-reset order, rollback logic, threshold, or acceptance rule changed. The metric is no longer silently omitted from the claim boundary.

## E — Scope and completeness claims: PASS

The main report remains accurately bounded: two failed 50-frame candidates, zero accepted attempts, zero committed frames, zero closed-loop PPO updates, and incomplete research scope at reports/PhyGile课程评估与262D闭环补全报告.md:44-62. Its audit note is correctly same-family and provisional at line 74.

The prior follow-up warning about stale README text is resolved. reimplementation/README.md:83 now describes the real replay adapter, fresh-process reconstruction, accepted-history replay, deterministic settings, and preserved failures. Line 85 accurately records the random-weight diffusion→262D decode→71D tracking→pre-reset feedback→reject/retry path, both candidates failing at 1/50 frames, 0 acceptance, 0 committed frames, the two one-frame proxy windows, and the absence of semantic training/closed-loop PPO adaptation.

## F — Evaluation type and recomputation: PASS

The revisions introduce no experiment or metric. Round-1 classifications and recomputations carry forward: curriculum, expert-mask, training-hook, and long-candidate execution are simulation_only; codec and held-pose checks are self_supervised_proxy; no real_gt or human_eval exists. The final manifest changed only the authorized documentation/release records. Reported numerical values, receipts, executed-source identities, raw arrays, checkpoint accounting, rollback deltas, terminal invariants, and the 0/2 long-candidate outcome remain bound to unchanged round-1 hashes.

## Action-item status and claim impact

| Item | Status | Impact |
|---|---|---|
| Publish completion archive sizes and SHA-256 | RESOLVED | Static manifest and SHA file match all four assets. |
| Aggregate MPJVE or label/remove it as diagnostic | RESOLVED_BY_EXPLICIT_LABEL | Promotion cannot be claimed to include velocity error. |
| Preserve engineering-only/proxy qualifiers | RESOLVED | Main report and implementation README preserve all material limits. |
| Correct stale README scope text | RESOLVED | README now matches the real adapter and failed 0/2 loop. |
| Obtain a different-family final review | OPEN, nonblocking | Acceptance remains provisional. |

**Blocking defects:** none.

The supported claim is engineering-interface validation with simulation/proxy evidence. Generated-motion success, stable terminal standing, semantic quality, author equivalence, learned GMT adaptation, paper-scale performance, and hardware performance remain unsupported.
