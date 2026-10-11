### env: closd-baseline-20261010 (warm runtime)
Provider: ssh ouyanghao@10.147.17.32, key C:/Users/Pende/.ssh/id_ed25519.
Spec: configs/env-closd-baseline-20261010.json. No package builds or installs.
Reuse frozen Python3.8.20/Torch1.13.1+cu117/IsaacGym Preview4 and closd-r3 overlay.
Source pinned d8950bdb78312fcd2844f336b3fc079e7368c87f; cached licensed assets de7106b947b6f70700b5320d1cd61fef4a9ebc9b.
GPU0; CPU threads2; system disk11GiB/data47GiB free at preflight.
New inference/recording wrapper requires independent document-following witness before ready.
Invocation and exact expected sentinel: docs/closd_baseline_20261010.md.

Validated: fresh doc-following agent executed --check once, exit0; seeded GPU witness and actual315-step4env recording PASS,4saved/0termination. receipt sha256 dff7d8e7334c56587eb3b3e7378962f2779cb20cd5c47c00eb3dc43201fdc165. No package changes. Full pipeline launched in tmux closd_baseline_20261010; completion not yet claimed.

Canonical env-spec SHA256: 81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780

2026-10-11 gap diagnostic v2 warm-reuse: fresh document-following witness exit0; GPU-kernel and actual4env299-step first-attempt recorder PASS. NPZ sha25628d1d425f00cff9bdc99e8ad2ae278bb5c6c726ea8f3379fa4a405420c82cbf3. Exactly one reward-internal reference-cache read per step; paired poses/contact finite. No native source/model/policy/environment changes. New invocation docs/gap_diagnostic_20261011.md; protocol3seeds x64envs. Formal experiment not yet claimed completed.

CORRECTION: v2 tracking conclusions are INVALID. The inherited reward reads an independent dataset reference, not the generated reference supplied to policy observations. Preserve v2 files and INVALID_V2_TRACKING.txt; exclude their tracking analysis. No model or policy failure can be inferred from that mismatch.

2026-10-11 v3 fresh document-following witness PASS: --check executed exactly once, exit0; all three sentinels present. 4env x299frames, all first attempts retained, no termination. NPZ SHA2567072a0303d31c82af973043fba336e4f247b1781e1055578f592204b27ec54f3, 669027 bytes. Generated observation ref_body_pos is copied BEFORE action and refreshed actual state AFTER the same physics step, before next observation. Coordinate/body-order checks PASS; all four references change on298 successive steps. Boundaries [0,1,61,121,181,241]; initial duplicate excluded by frozen first30frame rule. Canonical env/source unchanged; no installs/training. Witness does not validate task semantics, robot hardware or downstream utility. --run authorized after this witness.
