# Public-source scope for final generator acceptance

Original BABEL-paired index8291; selected officialSMPL-Xexport misses13 known
CMUsources; intended available profile8278. Author-private403clips remain
unavailable. This profile also does not claim all12296public tracking clips
from the paper. paper-step2.json now pins the original index and missing-index
aggregate SHA values and sets paper_data_scope_exactfalse. Per-source lists,
labels and retargeted arrays remain private.

The previous readiness field missing_public_sources=0 was ambiguous: it could
block the declared public profile forever or silently claim that13unavailable
sources existed. It is replaced by missing_available_public_sources=0 and a
separate, SHA-bound explicit source-scope artifact. No available motion can be
skipped just by calling it unavailable.

Acceptance artifact source_scope points to a JSON document with schema
textop.public_source_scope.v1,matchingprofile_id,original_index{path,sha256},
unavailable_index{path,sha256},prepared_manifests[{path,sha256},...]. These files
are private and read on the server. source_scope.py checks pinned hashes and
counts, normalizes only already-declared exact aliases, preserves source train/
val membership, rejects duplicate/foreign prepared sources, and requires the
prepared set EXACTLY equals requested minus the fixed unavailable set. It then
requires actual two-round decisions for every available id, including rejected
motions, against the verified dynamic-filter evidence. It never infers physical
success from preprocessing or source coverage.

check_step2_ready.py additionally requires accepted_corpus_loader_and_cache_
passed and a SHA-bound accepted_corpus_witness whose dataset_acceptedtrue and
candidate_view_onlyfalse. Unfiltered candidate-loader diagnostics cannot pass
that gate. Other actual filter, source-disjoint split, train-only statistics,
CLIP, runtime, provenance and resource checks remain necessary. Authorization
to auto-start after acceptance is preserved; no extra human approval is added.

10synthetic scope tests passed on serverPython3.8.10,actualexit0. Cases include
explicit declared gaps, missing available preparations/decisions, changed index,
duplicate/cross-split and foreign sources, silent gap-count changes, invented
private clips and false exact-paper-data claims. They are bookkeeping tests,
not actual corpus acceptance or physical filtering. Public proof is in
results/source_scope_20261025; private inputs persist under personal server
data_preparation_20261025/source_scope_inputs_v1 and original localD directory.
