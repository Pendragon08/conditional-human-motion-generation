# Independent preprocessing-backup document witness

The read-only audit executed successfully, actual exit 0. All 22 independent aggregate checks passed. Preprocessing remains INCOMPLETE: 2296 of 8278 available sources verified, 5982 remaining; pending cmu1330, bmlrub and kit. This matches the documented author counts and status with no observed divergence.

## One exact documented execution

Read PREPROCESSING_BACKUP_AUDIT.md and executed its exact PowerShell command once unchanged with --run document_v1. Local session 82344 returned exit 0. stdout's final aggregate JSON matches the saved receipt:
PREPROCESSING_INCOMPLETE_AVAILABLE_SOURCES_PENDING, preprocessing_complete=false.

The command and local exit proof are retained. Receipt elapsed time=16.95400000002701 seconds, NumPy=2.3.5. The source and spec were SHA256-hashed only, without reading implementation content.

The audit internally read licensed persistent archives as authorized. This independent witness inspected only the public aggregate receipt and permitted hashes. No manual archive opening, private source list/manifest/array copies, extraction, rebuilding, deletion, edits, retries, repairs, SSH commands, worker/waiter actions or publication occurred. Running GMR/FK and local waiters were not acted on.

## Actual archive/cohort evidence

| Cohort | Verified original sources | Mirrored sequences | Per-motion SHA verified | Largest selected69-field error |
|---|---:|---:|---:|---:|
| available500 | 500 | 500 | 500 | 1.1920910480966995e-7 |
| available143 | 143 | 143 | 143 | 1.1918107034603054e-7 |
| local1653 | 1653 | 1653 | 1653 | 1.1920904485762662e-7 |

All three cohort statuses are PERSISTENT_PREPROCESSING_BACKUP_AUDIT_PASSED. Each SHA-verification count equals its source count; each selected-field error is below 1e-5. Source train/validation totals are 1721/575, and every cohort's split counts sum to its source count.

The verified source count equals the sum of all passed cohorts. Remaining sources equal 8278 minus that actual sum. The check is based on observed passed archives and pending cohorts; it does not force a 2296 fixture or ignore a legitimately completed later backup. This run observed three passed and three pending cohorts.

Aggregate payload metadata: 2296 originals plus 2296 mirrors, 3,290,290 pose frames including mirrors, 34,542 paired annotations including mirrors, 126 rejected annotations retained (122+0+4), and 1,715,694,670 archive bytes. No motion was removed or physically accepted by this audit.

Archive hashes and detailed per-cohort metadata are retained in public receipt.json. They are the authorized audit program's verification evidence; this witness did not independently open archives or inspect private arrays/manifests.

## Hash checks

Computed source SHA256 matches the document and public receipt:
a4a9418aa2ee1e6730a4f113f2eecb1955081efe1b5f467147ab99a3d69e076e

Computed declarative-spec SHA256 matches the document and public receipt:
1dd2ba137f99bd173b8158dda2d1b8db27ed48ac2bdbd3be0a72e571e90cfa25

The program's documented fixed-index/source membership checks are carried by its passing cohort results and aggregate receipt. The missing/private index was not opened by this witness.

## Completeness and scope flags

read_only=true, output_writers_closed=true, available_target_sources=8278, unavailable_index_sources=13, author_private_clips_unavailable=403 and paper_data_scope_exact=false.

With pending cohorts, preprocessing_complete=false and the INCOMPLETE status are correct. Only all six verified archives with exact available-source/split coverage could establish completion of the available preprocessing profile.

dataset_accepted=false, physical_filter_executed=false and full_generator_training_started=false. This is a CPU/data audit with no GPU workload, optimizer, ML training or dynamic tracking. Recomputed named69D fields do not establish a new full FK/codec-equivalence proof. Filtered-data quality, full author-data reproduction and model readiness are not established.

## Saved public evidence

- documented-command.ps1
- stdout.log
- local-execution-receipts.jsonl
- receipt.json
- source_and_spec_sha256.json
- verification.json
- evidence-readback-receipts.jsonl
- REPORT.md
