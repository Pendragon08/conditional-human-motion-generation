## V55 audit addendum

### Revised status

- **Overall verdict remains FAIL** because the frozen V51 provenance failure is unchanged.
- **C changes from FAIL to PASS.** The remote V54 invocation log now exists and matches the local copy: `575a90ecdc6c7bd57aa292c6d611a3f3802c3c8d6dfc607d68735d3606e3b28d`.
- **Current selected-construction lineage: PASS with qualifications.**
- **Historical V54 selection-binding gap remains recorded.** V55 supplies separate current evidence; it does not rewrite or retroactively complete V54.

Revised A–F statuses:

| Area | Revised status |
|---|---|
| A. Ground-truth provenance | **FAIL**, unchanged |
| B. Score normalization | **PASS**, unchanged |
| C. Existence and counts | **PASS** |
| D. Dead code / invocation | **WARN**, unchanged |
| E. Scope | **WARN**, unchanged |
| F. Classification | **PASS**, unchanged |

### V55 artifacts

Local and remote hashes are identical:

- `D:\具身智能\conditional-human-motion-generation\scripts\verify_selected_construction_v55_20261012.py` — SHA-256 `d22c6ce574d458c47f83cb62160ecbca72b66ab28e6a09b7689b8da72897f42d`
- `D:\具身智能\conditional-human-motion-generation\results\selected_construction_binding_v55_20261012\certificate_1791770401890678522.json` — SHA-256 `7800698a34fdcb58710c81afc0ad17a2814c572b4fd6ffb826081d2939a685b0`

All 84 certificate input hashes currently match the remote files. The certificate contains six arm/seed entries, all with `exact_reference_reconstruction=true` and `selection_labels_reconstructed_from_bound_raw=true`.

### Does V55 close the current selection-label gap?

**Yes, for current-state evidence.**

V55 uses the correct trust direction:

- Candidate raw directories are passed through `guard_batch` at source lines `32-42`, binding their metadata, metrics, and trajectories to execution records.
- Candidate index `k` is recomputed using the frozen ranker at lines `51-53`.
- `construction_complete` is taken from the selected execution-bound candidate metric at lines `53,58-59`.
- The selected reference is rebuilt directly from the bound candidate `actual` trajectory and valid mask at lines `54-57`.
- The rebuilt reference and original goal are compared exactly with the selected NPZ at lines `61-63`.
- That NPZ’s motion hash must equal the replay request’s execution-bound `replay_motion` at lines `45-48`.
- The mutable selected JSON is read only for an equality check after reconstruction at lines `43-44,60`; `selected/*.lineage.json` is never trusted.

The reconstructed totals are:

| Arm | Seed 11 | Seed 12 | Seed 13 | Total construction complete | End-to-end joint |
|---|---:|---:|---:|---:|---:|
| Feedback | 72 | 72 | 70 | 214 | 0 |
| Random | 72 | 71 | 70 | 213 | 0 |

These match V53/V54. Current construction labels and selected motions are therefore independently supported by execution-bound candidate artifacts and replay-bound motion.

### Remaining limits

1. V55 is explicitly a **current reconstruction**, not proof that the original selected JSON had an immutable execution-time hash. Source line `73` and certificate line `155` preserve this limitation.
2. The V54 result does not bind the V55 certificate. The two artifacts must be cited together; V54 alone retains its historical selection-JSON gap.
3. Most checks use `assert`, and neither the certificate nor a V55 invocation receipt records `sys.flags.optimize == 0`. The certificate is not fail-closed against a hypothetical `python -O` invocation.
4. Certificate `input_hashes` omits the request files, config, `versions.json`, and the core/queue source files, although the running source checks several of them at lines `22-28`.
5. V55 does not reassert that all three candidates share `original_goal_sha256`.
6. End-to-end outcomes are combined with replay rows by positional `zip` at lines `64-66`, without an explicit `case_id` bijection. This cannot affect the present result because every replay `joint_pass` is false, but it should be fixed before any nonzero-result use.
7. Family composition, exact offsets, and custom-versus-official split scope remain outside V55.
8. V55 source and certificate remain Git-untracked; the remote/local HEAD mismatch remains.

### Revised reason codes

- `C_REMOTE_V54_INVOCATION_LOG_MISSING` — **RESOLVED**
- `V55_CURRENT_SELECTION_RECONSTRUCTED_FROM_BOUND_RAW` — **PASS**
- `V54_SELECTION_JSON_NOT_EXECUTION_BOUND` — **RETAINED AS HISTORICAL**
- `V55_NOT_BOUND_INTO_V54_ANALYSIS` — **WARN**
- `V55_ASSERT_OPTIMIZATION_UNATTESTED` — **WARN**
- `V55_REPLAY_ROWS_NOT_JOINED_BY_CASE_ID` — **WARN**
- `V55_DEPENDENCY_HASHES_NOT_EMITTED` — **WARN**
- `A_V51_SYNTHETIC_REFERENCE_NOT_PROXY_LABELED` — **FAIL, unchanged**
- `GIT_REMOTE_HEAD_MISMATCH` — **WARN, unchanged**

### Claim and action changes

The current construction-complete totals `214/213` and end-to-end joint totals `0/0` may now be reported as independently reconstructed from bound raw candidates and replay-bound motions. This does not support efficacy, acceptance, innovation, semantic quality, or real-robot executability.

The prior action to create a reconstruction is satisfied by V55. Remaining actions are to bind the V55 certificate into a new append-only analysis manifest, record optimization mode and an invocation receipt, emit all dependency/request hashes, join replay outcomes by `case_id`, assert common original goals and exact scope composition, and archive or commit the authoritative remote working state.
