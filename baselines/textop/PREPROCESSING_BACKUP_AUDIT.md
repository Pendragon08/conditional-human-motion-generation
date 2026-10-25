# Persistent preprocessing backup completion audit

This is a read-only local data audit, not a new training environment or GPU
workload. It reuses bundled Python/NumPy2.3.5 without package installation.
Declarative configs/preprocessing-completion-audit.json was written before
execution. It reads existing privateDarchives without extracting or copying
arrays, and writes only aggregate public receipts. Per-source identifiers and
failure details remain private under localD/datasets/preprocessing_audit_private_20261025.
No cleanup, checkpoint changes, SSH commands or worker control occurs.

Author_v1 actually exited0 in17.5s. It reread all2296original+mirror payloads
from3complete persistent archives, verified every NPZ SHA against the archived
manifest, source/split membership against fixedBABELindex minus13unavailable
sources, finite arrays/dimensions,50Hz timelines, unit quaternions, binary
contacts,69D contact/height/joint/difference fields, mirrored root and annotation
bounds/timestamps. Largest selected-field discrepancy1.1921e-7. All122+4 rejected
annotations remain recorded; no motion was removed or physically accepted.
Current report correctly states INCOMPLETE because CMU/BMLrub/KIT backup gates
are pending. Original algorithm source/constraints unchanged. This is not a
new full FK/69D-codec equivalence proof: only the named fields are recomputed.

## Fresh invocation once in PowerShell, unchanged

```powershell
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/audit_preprocessing_backups.py' --run document_v1
```

Output must not already exist. No edits, source inspection to repair, retries,
private-array copying, deletion, archive rebuilding, worker actions or publication.
Read the aggregate JSON at baselines/textop/results/preprocessing_backup_audit_20261025/document_v1/receipt.json.
Require directexit0,read_onlytrue,output_writers_closedtrue,dataset_acceptedfalse,
physical_filter_executedfalse,full_generator_training_startedfalse,
available_target_sources8278,unavailable_index_sources13,privateunavailable403,
paper_data_scope_exactfalse. Three static archives available500/143/local1653
must pass500/143/1653 each with per_motion_sha_verified equal source counts;
selected-fielderror<1e-5. Verifiedsource count must equal sum of passed cohorts;
remaining count must equal8278minusverified. Independentelapsed values may differ.

SourceSHAa4a9418aa2ee1e6730a4f113f2eecb1955081efe1b5f467147ab99a3d69e076e
SpecSHA1dd2ba137f99bd173b8158dda2d1b8db27ed48ac2bdbd3be0a72e571e90cfa25
Check hashes without reading implementation. Source and missing-index SHA are
checked internally against fixed paper-step2 data_scope. Do not open/copy raw
archives, per-source manifests or generated private source lists yourself.

With any pending cohort, preprocessing_complete must befalse and statusmustbe
PREPROCESSING_INCOMPLETE_AVAILABLE_SOURCES_PENDING. Current2296verified means
5982remaining, pendingCMU/BMLrub/KIT. If a real later backup completed during
this test, count/status must follow its actual verified archive, not a forced
2296fixture. Only all six verified archives and exact8278source/split coverage
can produce PREPROCESSING_ALL_AVAILABLE_SOURCES_BACKED_UP_VERIFIED_FILTER_PENDING.
Even that status is preprocessing completion only; physical_filter_executed,
dataset_accepted and full_generator_training_started remainfalse. This audit
never launches filtering/training or sends a notification itself.

Save command/stdout, public receipt and an independent REPORT.md under
work/textop_reproduction_20261024/preprocessing_backup_doc_witness_v1.
Retain observed failures. No messaging other chats, no publishing or
claiming filtered-data/model readiness. Existing GMR/FK queues remain intact.
