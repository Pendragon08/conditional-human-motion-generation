# Response to the provisional audit

The original audit MD/JSON are unchanged. The Chinese report now explicitly states length-ordered selection, absent per-sample IDs, exposure of the eight test clips, one-seed scope, and the distinction between author-side remote verification and fresh-reviewer availability. The report preserves WARN and all original numeric values; no training rerun or parameter change was made.

Publisher tracker LFS metadata is additionally preserved in ../hf_tracker_publisher_metadata.json. This post-audit artifact is not treated as a changed independent verdict. The publisher does not provide signed dataset checksums. Future evaluation should log return_keys and freeze a validation/final-test protocol.

The final report appends the audit limitation section after the review. The reviewed input hashes remain in EXPERIMENT_AUDIT.json; the final GitHub delivery certificate will cover the final report bytes.
