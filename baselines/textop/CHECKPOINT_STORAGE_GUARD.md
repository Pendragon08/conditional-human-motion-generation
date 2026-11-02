# Formal filter-policy checkpoint storage guard

This standard-library CPU witness checks only storage rules. No SDK/GPU/model
is imported, checkpoint created, dataset read, deletion performed or worker
stopped. Warm training dependencies and optimization math remain unchanged.

The filter-policy run must fit nine128MiB checkpoint bounds, a2GiB disk reserve
and512MiB log allowance before starting. This is a conservative checkpoint
bound, not a measured exact full-library checkpoint size. Recheck the personal
persistent writable destination before every save. Each100iterations also
checks the remaining budget. If external disk usage consumes it, save an
emergency state only if its own reserve check passes, then stop. If even that
does not fit, refuse the write and retain all preceding checkpoints. Actual
saved bytes exceeding the bound also stop continuation for a revised budget.
These checks cannot reserve space against concurrent writes by other users;
any I/O failure is preserved with its partial checkpoint.

After the author witness has passed, a fresh document agent runs exactly once:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/audit_checkpoint_storage_guard.py --run document_v1'
```

Inspect server receipts/checkpoint_storage_guard_document_v1/receipt.json.
Require directexit0, CHECKPOINT_STORAGE_GUARD_WITNESS_PASSED,11checks,
cpu_onlytrue,new_checkpoints_written0,training_startedfalse,
output_writers_closedtrue. The tests use live directory/mount/free-space checks
and simulated capacity boundaries, outside-root, RAM, symlink, ownership and
permission failures. They leave their tiny receipt and test symlink in that
unique directory; no existing files are changed. Compare source hashes with
the author receipt without reading implementation to repair. Do not retry,
modify code, launch training, change packages, publish or copy private data.
Save the stdout, receipt and concise independent REPORT.md under local
work/textop_reproduction_20261024/checkpoint_storage_guard_doc_witness_v1.

This does not validate8192 full-library GPU capacity, physical acceptance,
checkpoint tensor serialization or full training. Existing exact model/Adam/
normalizer/EMA/count restoration evidence remains separate and unchanged.
