---
language:
- en
license: other
size_categories:
- 10K<n<100K
pretty_name: TEXEDO Dataset
task_categories:
- robotics
tags:
- text-to-motion
- human-motion
- motion-generation
- amass
- claw
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

<h1 align="center" style="font-size: 1.6em;">TEXEDO Dataset</h1>

<p align="center" style="font-size: 1.6em; font-weight: bold;">Test-Time Scaling for Controller-Aware Language-Conditioned Humanoid Motion Generation</p>

<p align="center">
  <a href="https://jianuocao.github.io/TEXEDO/"><img src="https://img.shields.io/badge/Website-TEXEDO-blue" alt="Website"></a>
  <a href="https://github.com/JianuoCao/TEXEDO"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="Code"></a>
  <a href="https://arxiv.org/abs/2606.22998"><img src="https://img.shields.io/badge/arXiv-2606.22998-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/JianuoCao/TEXEDO-Checkpoint"><img src="https://img.shields.io/badge/Models-Hugging%20Face-yellow?logo=huggingface" alt="Models"></a>
</p>

TEXEDO is a text-motion dataset for the **Unitree G1 humanoid**, prepared from two sources: AMASS and CLAW. It keeps the MotionGPT-style split files while adding JSONL index files that are easier to load from the Hugging Face Hub. The dataset accompanies the TEXEDO paper, a text-to-motion pipeline that performs test-time scaling — sampling multiple candidate motions from a language prompt and selecting the best with controller-aware dynamic and semantic verifiers.

- 🌐 **Project page:** https://jianuocao.github.io/TEXEDO/
- 💻 **Code:** https://github.com/JianuoCao/TEXEDO
- 📄 **Paper:** https://arxiv.org/abs/2606.22998
- 📦 **Checkpoints:** https://huggingface.co/JianuoCao/TEXEDO-Checkpoint

## Dataset Structure

```text
TEXEDO_dataset/
  README.md
  train.txt
  val.txt
  test.txt
  data/
    train.jsonl
    validation.jsonl
    test.jsonl
    all.jsonl
  motions/
    amass/{id_prefix}/{id}.npy
    claw/{id_prefix}/{id}.npy
  texts/
    amass/{id_prefix}/{id}.txt
    claw/{id_prefix}/{id}.txt
  metadata/
    dataset_summary.json
  prepare_texedo_dataset.py
```

Each sample has one motion file and one text annotation file. Motion files are NumPy arrays with shape `(num_frames, 36)`.

## Motion Format

Each `.npy` motion file stores a float array with shape `(T, 36)`, where `T` is the number of frames. The 36 dimensions are organized as:

| Feature slice | Size | Description |
| --- | ---: | --- |
| `motion[:, 0:3]` | 3 | root position, `(x, y, z)` |
| `motion[:, 3:7]` | 4 | root quaternion, `(w, x, y, z)` |
| `motion[:, 7:36]` | 29 | joint positions / joint angles in the order below |

Joint order for `motion[:, 7:36]`:

| Joint index | Feature dim | Joint name |
| ---: | ---: | --- |
| 0 | 7 | `left_hip_pitch_joint` |
| 1 | 8 | `right_hip_pitch_joint` |
| 2 | 9 | `waist_yaw_joint` |
| 3 | 10 | `left_hip_roll_joint` |
| 4 | 11 | `right_hip_roll_joint` |
| 5 | 12 | `waist_roll_joint` |
| 6 | 13 | `left_hip_yaw_joint` |
| 7 | 14 | `right_hip_yaw_joint` |
| 8 | 15 | `waist_pitch_joint` |
| 9 | 16 | `left_knee_joint` |
| 10 | 17 | `right_knee_joint` |
| 11 | 18 | `left_shoulder_pitch_joint` |
| 12 | 19 | `right_shoulder_pitch_joint` |
| 13 | 20 | `left_ankle_pitch_joint` |
| 14 | 21 | `right_ankle_pitch_joint` |
| 15 | 22 | `left_shoulder_roll_joint` |
| 16 | 23 | `right_shoulder_roll_joint` |
| 17 | 24 | `left_ankle_roll_joint` |
| 18 | 25 | `right_ankle_roll_joint` |
| 19 | 26 | `left_shoulder_yaw_joint` |
| 20 | 27 | `right_shoulder_yaw_joint` |
| 21 | 28 | `left_elbow_joint` |
| 22 | 29 | `right_elbow_joint` |
| 23 | 30 | `left_wrist_roll_joint` |
| 24 | 31 | `right_wrist_roll_joint` |
| 25 | 32 | `left_wrist_pitch_joint` |
| 26 | 33 | `right_wrist_pitch_joint` |
| 27 | 34 | `left_wrist_yaw_joint` |
| 28 | 35 | `right_wrist_yaw_joint` |

## Splits

| Split | Samples |
| --- | ---: |
| train | 18,590 |
| validation | 2,324 |
| test | 2,325 |
| total | 23,239 |

## Sources

| Source | Samples |
| --- | ---: |
| AMASS | 9,245 |
| CLAW | 13,994 |

The original `textseedo` source label from the preparation metadata is normalized to `claw` in this release.

## JSONL Fields

Each row in `data/*.jsonl` contains:

- `id`: six-digit sample id
- `split`: `train`, `validation`, or `test`
- `source`: `amass` or `claw`
- `motion_path`: relative path to the `.npy` motion file
- `text_path`: relative path to the original text annotation file
- `num_frames`: number of motion frames
- `motion_dim`: motion feature dimension, currently 36
- `num_texts`: number of captions in the text file
- `captions`: parsed caption entries with `caption`, `tokens`, `start_time`, and `end_time`

This release intentionally does not include `raw_source` or `original_npz` fields.

## Loading

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download
import numpy as np
from pathlib import Path

repo_id = "JianuoCao/TEXEDO"
repo_root = Path(snapshot_download(repo_id, repo_type="dataset"))

ds = load_dataset(
    "json",
    data_files={
        "train": str(repo_root / "data/train.jsonl"),
        "validation": str(repo_root / "data/validation.jsonl"),
        "test": str(repo_root / "data/test.jsonl"),
    },
)

sample = ds["train"][0]
motion = np.load(repo_root / sample["motion_path"])
captions = sample["captions"]
```

## Citation

```bibtex
@misc{cao2026texedotesttime,
  title={TEXEDO: Test-Time Scaling for Controller-Aware Language-Conditioned Humanoid Motion Generation},
  author={Jianuo Cao and Yuxin Chen and Yuzhen Song and Masayoshi Tomizuka and Chenran Li and Thomas Tian},
  year={2026},
  eprint={2606.22998},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.22998},
}
```

## License & Terms

This release is distributed under the `other` license. The motions are derived from **AMASS** and **CLAW**; their original redistribution terms and licenses apply. Please review and comply with the source datasets' terms before use or redistribution.