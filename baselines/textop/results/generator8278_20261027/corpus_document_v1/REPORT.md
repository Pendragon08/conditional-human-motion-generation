# Generator8278 independent document witness

PASS — `GENERATOR8278_CORPUS_LOADER_WITNESS_PASSED`. Executed the documented command verbatim once; actual SSH, outer supervisor and child exits were all **0**, with no timeout. Outer elapsed time **55.69s**. **69/69** recorded checks passed; no document/runtime divergence or blocker found.

```text
python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_generator8278_corpus.py --mode verify --run corpus8278_document_v1
```

The unchanged warm environment has canonical hash `934d8221`. All three deployed corpus source/spec files matched local LF bytes before execution and remained unchanged afterward. GPU1 had 40,726 MiB free before launch; CPU affinity stayed 0–3 and Torch fraction was 0.65. Six resource samples observed a peak owned GPU allocation of 260 MiB; Torch tensor peak was 0 bytes because this verification read the existing cache.

All 8,278 preprocessed inputs were accounted for. Two TRAIN families (four variants) remain quarantined for exact validation pose duplication, leaving 8,276 families / 16,552 variants: 12,402 TRAIN and 4,150 VAL. A separate 12.05s CPU-only read independently checked original-index and six-manifest splits, reconstructed all stored float32 poses, and found **zero source-family overlap and zero exact-pose overlap**. Original/mirror pairs stay together, and all 2,075 validation families remain. No source identities or text arrays are included here.

All three real loader modes — uniform TRAIN, uniform VAL and weighted TRAIN — passed four primitives at `[4,512,10,69]`, with text `[512,512]`, zero missing embeddings and zero history-overlap error. Both sampler regressions passed. All eight actual cache artifact SHA256 values matched their receipt. The complete cache covers 93,632 annotation pairs and 7,623 unique CLIP strings.

TRAIN-only normalization provenance, boundary exclusion, finite mean/std and the 7,014,270 eligible TRAIN-feature count were checked. This independent pass verified the saved statistics SHA and frame accounting; it did not refit or numerically recompute all statistics.

Writers closed, the child was reaped, both worker processes exited, and the common lock was free. No cache build/copy, optimizer update, SDK execution, physical filtering or training was performed. This is acceptance of the user-authorized **pre-filter corpus loader**, not paper-filtered data, neural-gradient, physics or semantic-quality acceptance.

Receipt SHA256: `fa4a31d666ea085a541186814d2c1adc2b24e50801bc06a94627a1118e8e82fa`. View receipt SHA256: `bf66fac213d6b9a1812d0c1d578fe09599346e4a2211e70233b968743cd60113`. Exact source/spec and all artifact hashes are in `independent_summary.json`.

Full private logs, stderr, actual exit records, resource samples, read-only audit code/results and source snapshots: `D:\具身智能\TextOp复现_20261024\diagnostics\generator8278_corpus_document_v1`. Only this report and the aggregate JSON are published in the workspace.
