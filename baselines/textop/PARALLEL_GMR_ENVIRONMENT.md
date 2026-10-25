# Four-process GMR compatibility and timing witness

Warm GMR87f577cc, same declared packages/model/IK/robot assets, no installations.
The user authorized changing remaining preprocessing to parallel execution.
Four independent worker processes share fixed CPUs0,1,2,3; each preserves
the original four Torch/MKL threads and one NumPy/SciPy OpenBLAS thread. CUDA disabled. All workers and parent
remain within the same fourcores; this does not authorize sixteen CPU cores. Existing raw
upload/backup/FK workers and paused training remain intact. Only different
complete motions are parallel; each motion's frames still pass sequentially
through its original GMR instance. Canonical output/manifest/backup schema is
unchanged; worker scratch/manifests live outside the canonical archive folder.

Original kernel90db2afd is preserved under sources/gmr_serial_original_v1.
AST comparison verified unchanged algorithm functions and mainbody except a
cpu_threads argument/resource constant; CLI adds an explicit scoped dispatcher.
Author_v1 with one numerical thread perworker failed the unchanged1e-5
comparison (qpos difference0.00016665). Its outputs, failure and source snapshot
are retained; its speedup is NOT validated for production. Revision2 preserved all4thread numerical behavior and passed exact output
comparison, but measured only1.08x improvement. Revision3 keepsTorch/MKL4,
setsOpenBLAS1 but still measured only1.07x. Revision4 additionally makes idle
OpenMP helpers sleep (OMP_WAIT_POLICY=PASSIVE,KMP_BLOCKTIME=0), keeping the
same numerical kernels and precision gates. Author_v4 passed with zero array
difference and2.41x measured speedup; independent invocation below remains required.
The production gate is WAIT_FOR_VALIDATION until independent output checks pass.
Do not enable the gate or restart existing upload/queue workers yourself.

## Execute exactly once

Read configs/gmr-parallel-spec.json and SERVER_NOTES.md first. Run once from
PowerShell, outputdocument_v1 must not exist. Do not repair, retry, change flags,
install dependencies, stop other processes or run production GMR. Command:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-3 /data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/gmr_parallel_witness.py --run document_v1'
```

The driver first processes8 complete fixed-quantile CMU motions serially, then
the same8 using4 disjoint workers. It compares both against previous stored
CMU output. Unrestricted arrays/annotations are NOT generated or published;
all licensed motion, labels and per-source information stay in privateRAM.
Only public aggregate receipts and logs under serverrootparallel_validation/
document_v1 may be read/copied. Serverroot is
/data/code/ouyanghao/oyh/textop-reproduction-20261024.

Inspect receipt.json, serial.log, parallel.log. Require directexit0, status
TEXTOP_PARALLEL_GMR_WITNESS_PASSED, two children actualexit0, sourcecount8 and
zero failures, parallel worker_exit_codes[0,0,0,0],8 canonical outputs,
112array comparisons and144metadatafield comparisons, source sets disjoint,
contacts/timestamps/annotations exact, all floats within declaredatol1e-5 and
rtol1e-6, previous-kernel comparison passed. Source+spec SHA must match below.
Actual sampled worker affinity must be CPUs0-3 for each of4workerpids and all
job-affinities staywithin0-3; shared_four_core_worker_affinity_verified=true. Report both walltimes and observedspeedup even
if improvement is small; no guaranteed4x or full-cohort timing claim.
Require physical_filter_executedfalse,dataset_acceptedfalse,
full_generator_training_startedfalse,outputs_privatetrue,output_writers_closedtrue.
Diagnostic8-source outputs must NOT increase full-corpus production counts.

Source SHA:
- scripts/prepare_paper_data.py 06539b1a7593ef45beda815d07252b4850f2c47237f2d0000c77bdcd60f101a5
- scripts/parallel_prepare_data.py 79d532acd48f9ff3db2ea9096a2b6f211666c9cb3b1581b1ea3d37f49d88ec47
- scripts/gmr_parallel_witness.py a0ae6fc5e4dd6570219ed723e21c25f02cfa8e5d451a262775171229c4ca16a0
- configs/gmr-parallel-spec.json b797d14305c7459a926573033eedd355e506a9722e1db0dec6d6fa063198a657
- configs/data-prep-env-spec.json 6e7f21352c2839d62498a068103cfedcd3949f5416b1ca075477f6bdbef94591

Save command/stdout, receipt/logs and independent REPORT.md in
work/textop_reproduction_20261024/parallel_gmr_doc_witness_v1. Do not copy
private arrays/manifests/checkpoints or publish/message other chats. Report
failures and doc-versus-reality divergence to the parent, without a retry.
