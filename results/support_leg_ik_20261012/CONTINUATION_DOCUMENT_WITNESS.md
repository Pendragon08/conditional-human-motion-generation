# Independent continuation document witness

Status: FAIL

The documented CPU --check executed exactly once. Actual exit: 0. Sentinel: SUPPORT_LEG_IK_CONTINUATION_PREFLIGHT_PASS. No --run, retry, simulator, generation, training, install, API, source change or threshold change.

Original DOCUMENT_WITNESS.json/.md remain FAIL and are hash-bound unchanged. Separate scalar diagnosis confirms documented preparation invariants; exact logged float64 norm equality is not a documented requirement. Historical scalar reduction context remains unproven.

preflight_sha256: 4f9ad0848212d465396004f7c6b75f66dddafd399fecdc8fdfea1b5f44d6d4b9
source_sha256: d607f7e1d41f3539cdb9e1cce5241ebdca1315829d3a702b86e7f9445f4a41a2
document_sha256: 080ff5a4ad30ec7bb8f1bfc4dfc99be56970761edc475c1561874f1a87a2b190

Independent saved-array geometry:
{
  "baseline": {
    "max_joint_shift_m": 0.0,
    "max_segment_error_m": 0.0,
    "max_paired_feet_float32_roundoff_m": 0.0
  },
  "cartesian": {
    "max_joint_shift_m": 0.0782050916595791,
    "max_segment_error_m": 0.05798189353482597,
    "max_paired_feet_float32_roundoff_m": 5.960464477539063e-08
  },
  "two_link_ik": {
    "max_joint_shift_m": 0.0782050916595791,
    "max_segment_error_m": 8.361849629379137e-08,
    "max_paired_feet_float32_roundoff_m": 5.960464477539063e-08
  }
}

Limitations:
- One-seed internal-validation preparation integrity only; no efficacy or novelty conclusion.
- Cartesian control may violate bone lengths; it is a positional representation ablation.
- Classical positional IK supplies no joint-limit, torque, collision or full-dynamics guarantee.
- Endpoint reference equality does not establish actual endpoint success.
- Historical scalar reduction context remains unproven; logged low-bit norm discrepancies remain retained.
- Original C2/C3 gates remain in force.

All inspected hashes, checks, immutable lineage, cached charge and failures appear in JSON.
Failures: [{"check": "cached_parent_budget_ledger.json", "detail": null}]
