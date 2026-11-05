# Remote publication coverage verification

Captured: 2026-11-05T11:07:57.372Z. Status: **remote_snapshot_uploaded_and_independently_verified**.

The release is still a draft. This is an upload verification, not a new experiment result.

| Asset | Files | Uploaded bytes | SHA256 | Verified |
|---|---:|---:|---|---|
| [dreamcontrol-v2-remote_code_snapshot-20261105-part01.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote_code_snapshot-20261105-part01.tar) | 2729 | 15623680 | 8aae22bf2d74e7b0e2d1658e792a0d8e593b6e8c4889b548f8967f5b6fb3366a | yes |
| [dreamcontrol-v2-remote_generated_reference_or_execution-20261105-part01.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote_generated_reference_or_execution-20261105-part01.tar) | 10087 | 861113856 | 9e3f86afc88a426eddabd71fdf7ad29c426e352cc3d2175d3a57603e0d12b710 | yes |
| [dreamcontrol-v2-remote_media-20261105-part01.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote_media-20261105-part01.tar) | 328 | 61617664 | e34bc787c1175abb765b6922129c1e7b5406ea78fa75d01db81eb704b7db3767 | yes |
| [dreamcontrol-v2-remote_report_config_log-20261105-part01.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote_report_config_log-20261105-part01.tar) | 11402 | 382739968 | 85799be08e861a0b56961ce9145c169676089e6a4eb71897cfb66348e165bfed | yes |
| [dreamcontrol-v2-remote-models-20261105-part01.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote-models-20261105-part01.tar) | 37 | 1057892352 | 82f5d013ad950de329591dc29d1f7ad517596447a0859e95317cb2bd72fc8cee | yes |
| [dreamcontrol-v2-remote-models-20261105-part02.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote-models-20261105-part02.tar) | 85 | 520439808 | 1c65264f98563e5bbaa9dbd91174e650e6800cec6bc7fe4cc542f42072e76cf2 | yes |
| [dreamcontrol-v2-remote-cold-models-20261105-part01.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote-cold-models-20261105-part01.tar) | 3 | 651594240 | 7fd0ff6f14a63ed926a7ea1b3d73bb84889711ec56e0e92d4671471d7b9ccc3f | yes |
| [dreamcontrol-v2-remote-cold-models-20261105-part02.tar](https://github.com/Pendragon08/conditional-human-motion-generation/releases/download/untagged-fbed02e890b8c5d9c634/dreamcontrol-v2-remote-cold-models-20261105-part02.tar) | 1 | 486550016 | 624fc6853f51370c290f0b2ce63ee9db7514405d7ac6f0ba180bce57ec8be33b | yes |

Checkpoint coverage: 126/126 remote locations (122 direct model files and 4 root gzip archives). Missing: 0; content mismatches: 0; duplicate paths: 0; manifest discrepancies: 0.

Every included remote TAR manifest was compared to the fixed source plan by relative path, byte count, and SHA256. The GitHub asset ID/name/state/length/SHA256 was independently re-fetched and compared to the uploader receipt. Source file identity was verified by the streaming uploader before and after reading. No source was modified by this check.

The 138 locally archived checkpoints are uploaded separately and are not counted as completed in this remote-only receipt. Active/future Pour, bimanual-box, and push/pull checkpoints and evaluation reports are not certified by this snapshot. Raw dataset corpora, third-party pre-trained dependencies, and explicit active-file holds remain excluded.

To restore, download an asset, verify its complete SHA256, extract into an empty directory, then verify each file against `PUBLICATION_MANIFEST.json`. For root gzip model files, verify the compressed checksum first and the original checkpoint checksum after decompression. Never overwrite an existing different-content source.
