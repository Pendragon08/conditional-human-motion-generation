# C3 implementation and prospective validation boundary

The allocation core is implemented, not experimentally validated. It estimates
joint repair yield per newly paid candidate call from paired validation-only
fixed-reference replay attempts; construction-pass labels are not replay utility.
Calibration refuses missing repair observations, split leakage or a
complete lack of observed joint repairs. The unit tests use explicitly synthetic
fixtures, not experimental outcomes or physical ground truth.

Uniform, random, tracking-difficulty and proposed allocation all cover every
input with a paid first attempt, preserve every failed source, and enforce no
more than three paid attempts per input. Three generation seeds and all original
target constraints remain necessary. The core schedules future calls and never
counts a scored existing bank as reduced acquisition cost.

An important design distinction: at a full budget of 3N calls for N inputs,
every method exhausts all three attempts per input, so allocation counts cannot
differ. A prospective C3 comparison must therefore predefine a genuinely scarce
equal budget, such as 2N total calls, and record actual calls plus elapsed
generation/simulation cost. This differs from, and must not replace, C1's fixed
three-candidate experiment. The proposed 2N schedule is not yet launched.

Current V28 has no joint repair benefit and V33 shows a strong execution/proxy
floor on original references. Paired validation repair calibration has not been
collected for C3. Do not fit allocation utility from held-out test outcomes or
invent gains to bypass these prerequisites. Before efficacy runs, resolve the
measurement semantics, obtain informative validation repair observations, freeze
the allocator and budget, obtain a fresh document-following witness, then compare
actual newly acquired data and frozen-controller downstream outcomes.

```powershell
python scripts/repair_budget_core_v38_20261012.py
```

Expected `REPAIR_BUDGET_CORE_FIXTURES_PASS`; CPU-only functional check, not a
candidate efficacy test. No simulator, model, training, API or Overleaf writes.
