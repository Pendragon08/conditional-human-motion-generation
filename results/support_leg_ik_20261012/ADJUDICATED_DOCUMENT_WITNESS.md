# Independent adjudicated document witness

Status: PASS

The documented CPU --check executed exactly once. Actual exit: 0. Sentinel: SUPPORT_LEG_IK_ADJUDICATED_PREFLIGHT_PASS. No --run, retry, simulator, generation, training, install, API, source change or threshold change.

Original DOCUMENT_WITNESS.json/.md remain FAIL and are hash-bound unchanged. Both historical FAILs remain unchanged and hash-bound. Separate scalar and continuation adjudications confirm documented preparation invariants. Historical whole-ledger SHA is reconstructed from the unchanged baseline row; current append-only ledger equality is not required; exact logged float64 norm equality is not a documented requirement. Historical scalar reduction context remains unproven.

preflight_sha256: cc6402c1ff1908711acc5c93d78ee34040591e291ce59bbfeedd7795c1c64a83
source_sha256: e8e77e48d1ebb6d064d9fab2f99ac068be67d727ae4ed9dd21b5ba6c1cb44ead
document_sha256: acdbcb771c4ac6e9318fe70e7a9ddee976860000f89459cd4daac150cc81b4e3

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
Failures: []
