# 69D candidate-view loader diagnostic

Warm reuse of paper69 environment934d8221, source ledger SERVER_NOTES.md.
configs/candidate-loader-spec.json was written before dispatch. No package or
shared-source edits. GPU1/CPU4/Torch65%. Full build uses the already-completed
500+143+1653private GMR manifests (2296sources plus mirrors), not partial CMU
outputs and not an accepted dataset. No model gradients or optimizer is used.

Author build is running under supervise_candidate_loader.py --run
candidate_loader_author_v2 --build. Do not rebuild or duplicate its candidate
directory. After its actualexit0, fresh document invocation once, unchanged:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_candidate_loader.py --run candidate_loader_document_v1'
```

Require actualexit0/no timeout, TEXTOP_CANDIDATE_LOADER_SUPERVISED_EXIT0 and
TEXTOP_CANDIDATE69_LOADER_WITNESS_PASSED. Source_sequences2296examined,
variants4590after quarantining1TRAINsource family/2variants;validation unchanged,
0optimizerupdates,candidate_view_onlytrue,dataset_acceptedfalse,
physical_filter_executedfalse,full_generator_training_startedfalse. Three real
official-loader passes: uniformtrain,uniformval,weightedtrain;4primitives each
[512,10,69],text[512,512], history2overlaperror<1e-5, no missing text embeddings,
finite normalization. Exact35pose-frame and invalid-weighted-index regressions
must pass. Retain actual peakGPU/elapsed values without extrapolating model-
training or full8278source capacity. This validates the loader on2296unfiltered
sources, not the future accepted full corpus or complete-model batch512memory.

Builder fits float64 population statistics only on eligible TRAIN motion
frames+mirrors, floorstd1e-6, and computes frozen real CLIPViT-B/32 embeddings
for original and mirrored labels. Original body/text arrays and known source
splits are retained, source and exact pose hashes checked across splits. Uses
the pinned released action-category weights; neither exact author69D statistics
nor private category distribution is claimed. Full-frame population statistics
are an explicit convention instead of upstream's approximate random-window
estimate. Cross-source transitions are excluded before normalization and
equivalence checked on real source excerpts. They MUST be refitted after real
filtering, with a new accepted-view
fingerprint; candidate normalization/cache cannot establish training readiness.

Author_v1 stopped before fitting/export because a rebuilt full-source TRAIN
motion and a VAL motion (plus their mirrored forms) had exactly equal robot
poses. Private audit confirmed2pose groups,1train/1val source,275frames; raw
SMPL-X numeric arrays differ by at most9.6e-13,betas/gender/fps agree, despite
different file hashes. This is a near-identical source capture, not a static
retargeting collapse (joint range1.76rad,root range5.37m). V1failure/source/logs
are retained. V2quarantines both variants of the TRAINsource, keeps validation
and all authoritative files unchanged, records private exclusions and refits
TRAINstatistics afterward. This is an explicitly declared data-quality choice;
repeat the analysis on the final accepted corpus instead of blindly reusing
this provisional candidate exclusion. No files are deleted or relabeled.

paper69_data_runtime.py corrects two released sampler indexing defects in this
process only: sample within valid-source weights (not indexing a filtered list
with unfiltered ids), and randint's inclusive final legal start, allowing the
single valid window for exactly35pose frames. Official primitive extraction,
69D codec, CLIP text encoding and label-selection rule stay unchanged. Actual
network numerical validation remains in PAPER69_ENVIRONMENT.md.

Public receipts: generator_validation/candidate_loader_document_v1/receipt.json,
receipts/candidate_loader_candidate_loader_document_v1_exit.json and logs/
candidate_loader_candidate_loader_document_v1.log. Copy only these public
artifacts, not candidate PKLs, statistics/cache, private index/labels or source
arrays. No edits/retries/source inspection/repair, other process actions or
publication by witness. Save command/stdout and REPORT.md under
work/textop_reproduction_20261024/candidate_loader_doc_witness_v1. Check hashes
without inspecting source. A failure remains recorded; do not erase/rebuild the
view to force a pass. Existing GMR/FK queues and paused57D process stay intact.
