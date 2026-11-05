# Recovering the published reproduction artifacts

`src/restore_publication_archive_20261105.py` is a Python standard-library tool for downloaded GitHub release artifacts. It does not access credentials, download anything, launch training, or modify the source archives. Use Python 3.10 or newer.

The release is `Pendragon08/conditional-human-motion-generation`, tag `dreamcontrol-v2-reproduction-20261105`. Download the receipt and the assets it names from that release. Draft-upload receipts may contain temporary `untagged-…` URLs; use the final release asset names and published catalog when locating files. A download directory must contain the exact asset basenames in the receipt.

## What the checks establish

Before creating output files, the tool verifies every downloaded part's byte count and SHA256, then verifies the byte count and SHA256 of their ordered concatenation. It reads split files as one virtual stream, so a second 4.4 GB concatenated tar file is unnecessary. It verifies the tar member names and lengths against the embedded `MANIFEST.json` or `PUBLICATION_MANIFEST.json`. Each restored member is independently SHA256-checked before it becomes visible at its final name.

Absolute paths, `..`, Windows drive/stream names, archive symlinks, hardlinks, and special files are refused. Existing destination files are accepted only if their contents already match the manifest; differing contents are never overwritten. Output-parent symlinks are refused. The downloaded parts and archives remain intact. Only a temporary output file created by the current invocation is removed after publishing its verified contents or after an error.

Restoration uses an atomic hard link within the output directory to refuse concurrent name replacement. Use a filesystem with hard-link support, such as NTFS or ext4. A failure on a filesystem without that support does not overwrite existing data.

## Split historical checkpoint archive

First obtain the final, complete `archived-weights-v3-receipt.json` and every `archived-weights-v3.partNNNN` file listed in it. An incomplete or failed receipt is refused.

```console
python src/restore_publication_archive_20261105.py --receipt publication/archived-weights-v3-receipt.json --downloads DOWNLOAD_DIRECTORY --list
```

The listing includes the archive member, original recorded path, byte count, and content SHA. To restore one checkpoint, replace the example member below with its exact listed name:

```console
python src/restore_publication_archive_20261105.py --receipt publication/archived-weights-v3-receipt.json --downloads DOWNLOAD_DIRECTORY --member weights/EXACT_SHA256.pt --output RECOVERY_DIRECTORY
```

Use `--all` instead of `--member` only when all checkpoint files are needed. `--member` can be repeated. Restored checkpoint names follow the archive's `weights/<SHA256>.pt` layout. The tool does not write to the absolute historical server paths recorded in the manifest. The caller can subsequently arrange the verified files for a new machine; the manifest retains their provenance.

## Independent server-side tar archives

The eight remote assets are independent tar archives, not pieces to concatenate. Choose one by name from `remote_upload_receipt.json`:

```console
python src/restore_publication_archive_20261105.py --receipt publication/remote_upload_receipt.json --downloads DOWNLOAD_DIRECTORY --asset dreamcontrol-v2-remote-models-20261105-part01.tar --list
```

Use the same command with `--member EXACT_RELATIVE_MEMBER --output RECOVERY_DIRECTORY`, or with `--all --output RECOVERY_DIRECTORY`, to recover content. These tars retain relative source paths and contain an embedded `PUBLICATION_MANIFEST.json`. Source code, references, evaluation evidence, media, active-at-capture snapshots, and final weights remain distinct in the published inventory. A snapshot of an unfinished run is not a final completed result.

`local-code-reports-media-v2-part01.tar` works the same way with `local_results_upload_receipt.json`. Its content-addressed `files/<SHA256>` layout is mapped to original project paths by the embedded manifest. Repeated identical source files may share a content-addressed member name; the tool restores that content once while the original full manifest retains all path mappings.

## Completed logs and review images

The completed-log tar was preserved byte-for-byte from the earlier cold archive and has no embedded manifest. Use its published external manifest with the complete chunk receipt:

```console
python src/restore_publication_archive_20261105.py --receipt publication/completed-logs-review-v3-receipt.json --downloads DOWNLOAD_DIRECTORY --manifest publication/completed_logs_and_review_png_manifest.json --list
```

Then select exact members with `--member` and an explicit `--output` directory. The external manifest is the original `training_space_archive_20261105/completed_logs_and_review_png/v1/manifest.json`; its rows bind all 133 members to their historical paths, bytes, and SHA256 values. Publish this manifest alongside the receipt if using the above public layout.

## Space and validation

The tool checks free space for all selected, not-already-identical files before restoring content and retains a default reserve of **1 GiB + 100 MiB**. There is no staging tar. For a volume that requires a larger reserve, such as the project's server root volume, pass `--reserve-bytes 2252341248` for **2 GiB + 100 MiB**. Use a different destination volume when the check fails; the project's established storage floors remain in force.

Eight small temporary-file tests cover split-stream recovery, preservation of downloaded parts, identical-file reuse, refusal of differing content, corrupt and missing parts, incomplete receipts, path traversal, insufficient space, and source mutation after verification. They passed on Windows with the existing local Python environment:

```console
python -m unittest discover -s tests -p test_restore_publication_archive_20261105.py -v
```

No project checkpoints or datasets were read by these tests. The helper checks byte recovery, not scientific validity, policy success, or provenance beyond the supplied hash-bound manifests.
