# Independent Quick69 retained-reference corpus verification

The CURRENT instruction and exact documented verify invocation passed once on the unchanged warm paper69@934d8221 runtime. No source/config/dependency change or second build was performed.

44 independent checks passed. Actual supervisor/child exits were both 0, supervisor PID 4030434, child PID 4030441, elapsed 4.163s. No timeout, signal cleanup, error or remaining worker; writers closed and common lock released.

42 original references retained: TRAIN31/VAL11, 174 annotation pairs, 74 nonempty CLIP texts, 14505 TRAIN feature frames. All 42 saved motions exactly match prepared references and all 42 annotations match the original view. Source and exact float32-pose split overlap are both zero.

| Mode | Primitive feature shape | Text shape | History overlap error | Missing embedding | Max absolute normalized feature |
|---|---|---|---:|---:|---:|
| train uniform | 4 × 512 × 10 × 69 | 4 × 512 × 512 | 0.0 | 0 | 28.637094497681 |
| val uniform | 4 × 512 × 10 × 69 | 4 × 512 × 512 | 0.0 | 0 | 28.740633010864 |
| train weighted | 4 × 512 × 10 × 69 | 4 × 512 × 512 | 0.0 | 0 | 21.974920272827 |

History2/future8 and seed20261026 are unchanged. The values are observed max absolute magnitudes, not recorded signed minima/maxima. No clipping or normalization threshold change. All three results match the author run exactly.

Read-only cache audit: TRAIN61 keys (60 nonempty + empty sentinel), VAL20 keys (19 nonempty + sentinel); zero missing or unused keys, finite float32 512D embeddings. Union74 nonempty texts. Saved69D mean range [-0.465249478817,0.948362648487], std range [0.003586556762,0.599491477013], zero dimensions at/below1e-6 floor. Statistics were read, not refitted.

Actual logs show three cached text loads and three cached mean/std loads, zero computation messages. All9 corpus files (8 artifacts plus receipt) retain identical SHA/size/mtime, with all19 frozen files and source/spec/view hashes unchanged. Verification peak Torch allocation0B is consistent with cached CPU loading; author build peak442733568B. Separate cache audit disabled CUDA and did not initialize it.

GPU1 free40726MiB at launch (>20GiB), CPU0–3, Torch cap0.65. Old57D PID3237858 remains T stopped. No other process or shared dependency was changed.

Doc-vs-reality: no functional divergence for CURRENT/verify. The lower author command still names historical author_v1; CURRENT explicitly reports v3 and forbids build. Corpus metadata retains its original BUILT_LOADER_PENDING label because the corpus files are immutable; actual author/document validation receipts both pass.

Scope: loader/corpus verification only; 0 optimizer/SDK/physics, no model/checkpoint or training. quick_training_ready=false and whole_library_filtered=false. Quick trainer gradient/resume/storage acceptance and paper-scale acceptance remain pending.

Exact server evidence is in server_evidence/; all11 files were copied byte-for-byte and SHA-checked. Private NPZ/joblib/text/index data stayed on the server. Nothing was written to public BASE/results.

| Receipt | SHA256 |
|---|---|
| Corpus | d0a44b975c8f390f25dc5e95c414b3b8354dacacc19d5673d7a86e6d21c9c8f1 |
| Verify | bd2dd3eff0b74f101b2a246667dcd980de60ff6f5457f821409fd0b0929d16b9 |
| Supervisor exit | b51df29eec7ca1c1e14a8cd86eed0a7e6f73c76e98cb505ef8832d7757337c54 |
| Process exit | 261bf72dbe022d787c7e49dff7526de646bda3b5bc6ab48813b6b00b30912c02 |
| Resources | 67cfdfa018d7e05cc18bb8c444c5e3f910986141aa5f11a76a734ca980b1abf6 |
| Log | a133d4b629cbdf0d01d4f7b2b736b2a8a6f7685de38224dfdd1d6c8a2781c9ca |
