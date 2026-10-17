# Phygile

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://baojch.github.io/phygile-page/)
[[Video]](https://www.youtube.com/watch?v=vFGOVQhbUCo&feature=youtu.be)

## Overview

Phygile is a humanoid whole-body motion tracking framework supporting multiple training paradigms:

- **Single-motion MLP**: Train a dedicated policy for a single reference motion file using an MLP actor.
- **Multi-motion MLP with Curriculum**: Train a single MLP policy on a large motion dataset with curriculum learning, freeze/drop mechanisms, and adaptive sampling.
- **Fine-tuning**: Fine-tune a pre-trained multi-motion checkpoint on a smaller set of motions (no curriculum, no freeze).
- **Multi-motion MoE**: Train a unified policy via a Mixture-of-Experts actor with curriculum-constrained routing (Phase A) and dynamic expert expansion (Phase B).

## Installation

- Install Isaac Lab v2.1.0 by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
git clone git@github.com:HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files:

```bash
cd whole_body_tracking
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Install the library:

```bash
python -m pip install -e source/whole_body_tracking
pip install wandb==0.12.21
```

## Environment Setup

Before running any script, set up the environment:

```bash
conda activate bjc@beyondmimic
export WANDB_API_KEY=<your_wandb_api_key>
export CUDA_VISIBLE_DEVICES=0,1
export DISPLAY=:2
```

## Data Processing

Reference motions must be converted to `.npz` format with start/end interpolation via forward kinematics before training.

### Single File

```bash
python scripts/csv_to_npz.py \
  --input_file <path_to_motion.csv> \
  --input_fps 30 \
  --output_name <motion_name> \
  --headless
```

### Batch Processing (with start/end interpolation)

Process an entire folder of CSV/PKL motions into `.npz`:

```bash
python scripts/csv_to_npz_inter_batch.py \
  --input_folder <path_to_csv_folder> \
  --input_fps 30 --output_fps 50 \
  --prefix_name <dataset_prefix> \
  --output_path <output_npz_dir> \
  --no_render --no_wandb --headless
```

Examples for different datasets:

```bash
# LAFAN1
python scripts/csv_to_npz_inter_batch.py \
  --input_folder /ssd/humanoid/dataset/LAFAN1_Retargeting_Dataset/g1 \
  --input_fps 30 --output_fps 50 \
  --prefix_name lafan \
  --output_path /ssd/humanoid/dataset/bm_all \
  --no_render --no_wandb --headless

# HumanML3D
python scripts/csv_to_npz_inter_batch.py \
  --input_folder /ssd/humanoid/dataset/humanml_gmr/pkl/csv \
  --input_fps 30 --output_fps 50 \
  --output_path /ssd/humanoid/dataset/humanml_gmr/npz \
  --no_render --no_wandb --headless

# ACCAD
python scripts/csv_to_npz_inter_batch.py \
  --input_folder /ssd/humanoid/dataset/ACCAD_gmr_30bm/csv \
  --input_fps 30 --output_fps 50 \
  --prefix_name amass \
  --output_path /ssd/humanoid/dataset/subset_ACCAD \
  --no_render --no_wandb --headless

# MoCapAct (50fps input)
python scripts/csv_to_npz_inter_batch.py \
  --input_folder /ssd/humanoid/dataset/mocap_0128_add/pkl \
  --input_fps 50 --output_fps 30 \
  --prefix_name mocap \
  --output_path /ssd/humanoid/dataset/mocap_0128_add/npz/ \
  --no_render --no_wandb --headless

# bones_studio
python scripts/csv_to_npz_inter_batch_bones.py \
  --input_folder /ssd/humanoid/dataset/bones_studio/g1/csv \
  --input_fps 120 --output_fps 50 \
  --prefix_name "" \
  --output_path /ssd/humanoid/dataset/bones_studio/g1/npz \
  --recursive \
  --no_render --no_wandb --headless

python scripts/replay_npz.py --motion_file  /ssd/humanoid/dataset/bones_studio/g1/npz_test/body_check_001__A155_M.npz
```

### Motion Replay

Verify processed motions by replaying in Isaac Sim:

```bash
python scripts/replay_npz.py --motion_file <path_to_motion.npz>
python scripts/replay_npz.py --motion_file /ssd/humanoid/dataset/humanml_gmr/npz/001758.npz
```

## Training

### Single-Motion MLP Training

Train a dedicated MLP policy for a single reference motion file:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_future_conv_single.yaml \
  --motion_file <path_to_motion.npz> \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name <project_name> \
  --run_name <run_name> \
  --no_wandb_model_upload
```

For example, to train on a dance motion:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_future_conv_single.yaml \
  --motion_file /ssd/humanoid/dataset/humanml_gmr/npz/005113.npz \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name bmtest \
  --run_name dance \
  --no_wandb_model_upload
```

### Multi-Motion MLP with Curriculum Training

Train a single MLP policy on a large multi-motion dataset with curriculum learning (progressive difficulty levels), freeze/drop mechanisms (automatically freeze well-learned motions and drop persistently failing ones), and adaptive sampling:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_future_conv_large_freeze.yaml \
  --motion_dir <path_to_motion_dataset> \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name <project_name> \
  --run_name <run_name> \
  --no_wandb_model_upload
```

To resume from a previous checkpoint:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_future_conv_large_freeze.yaml \
  --motion_dir /ssd/humanoid/dataset/humanml_gmr/npz \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name bmtest \
  --run_name freeze_curriculum_lv1 \
  --no_wandb_model_upload \
  --resume True \
  --load_run <previous_run_name> \
  --checkpoint <checkpoint_name>.pt
```

### Fine-tuning

Fine-tune a pre-trained multi-motion checkpoint on a smaller set of motions. This mode disables both curriculum learning and freeze/drop, suitable for adapting a general policy to specific motions:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_finetune.yaml \
  --motion_dir <path_to_finetune_motions> \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name <project_name> \
  --run_name <run_name> \
  --no_wandb_model_upload \
  --resume True \
  --load_run <pretrained_run_name> \
  --checkpoint <checkpoint_name>.pt
```

### Multi-Motion MoE Training

MoE training is split into two phases. Phase A trains with curriculum-constrained routing on a large multi-motion dataset. Phase B continues from a Phase A checkpoint and enables dynamic expert expansion on new motions.

#### Phase A: Curriculum-Constrained Training

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_moe_phase_a.yaml \
  --motion_dir <path_to_motion_dataset> \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name <project_name> \
  --run_name <run_name> \
  --no_wandb_model_upload
```

To resume from a previous checkpoint:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_moe_phase_a.yaml \
  --motion_dir /ssd/humanoid/dataset/humanml_gmr/npz \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name bmtest \
  --run_name moe_phase_a_soft_top3_conlv7_211csv \
  --no_wandb_model_upload \
  --resume True \
  --load_run <previous_run_name> \
  --checkpoint <checkpoint_name>.pt
```

#### Phase B: Dynamic Expert Expansion

Start from a Phase A checkpoint and train on new motion datasets:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --g1_training_cfg source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/g1_training_moe_phase_b.yaml \
  --motion_dir <path_to_new_motion_dataset> \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --log_project_name <project_name> \
  --run_name <run_name> \
  --no_wandb_model_upload \
  --resume True \
  --load_run <phase_a_run_name> \
  --checkpoint <checkpoint_name>.pt
```

### Policy Evaluation & ONNX Export

Play a trained policy. The script automatically exports the policy to ONNX format (saved to `<log_dir>/exported/`) and copies the motion file alongside it:

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs 1 \
  --load_run <run_name> \
  --checkpoint <checkpoint_name>.pt \
  --motion_file <path_to_motion.npz> \
  --play_full_motion
```

For MoE Phase B models, add `--phase_b` to activate all experts and specify `--experiment_name`:

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs 1 \
  --experiment_name g1_flat_moe_phase_b \
  --load_run <run_name> \
  --checkpoint <checkpoint_name>.pt \
  --motion_file <path_to_motion.npz> \
  --play_full_motion \
  --phase_b
```

Headless ONNX export (no GUI):

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs 1 \
  --load_run <run_name> \
  --checkpoint <checkpoint_name>.pt \
  --motion_file <path_to_motion.npz> \
  --headless
```

### Batch Evaluation

Evaluate a trained policy on an entire motion folder:

```bash
python scripts/rsl_rl/eval_metrics_folder.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs 8000 \
  --device cuda:0 \
  --load_run <run_name> \
  --checkpoint <checkpoint_name>.pt \
  --motion_folder <path_to_motion_folder> \
  --headless \
  --eval_runs 30
```

## Code Structure

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/`** — MDP definition: commands, rewards, events, observations, terminations
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`** — Environment hyperparameters
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`** — PPO hyperparameters
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/cfg/`** — Training configs:
  - `g1_training_future_conv_single.yaml` — Single-motion MLP config
  - `g1_training_future_conv_large_freeze.yaml` — Multi-motion MLP with curriculum + freeze/drop
  - `g1_training_finetune.yaml` — Fine-tuning config (no curriculum, no freeze)
  - `g1_training_moe_phase_a.yaml` — MoE Phase A config
  - `g1_training_moe_phase_b.yaml` — MoE Phase B config
- **`source/whole_body_tracking/whole_body_tracking/utils/encoder_actor_critic.py`** — MoE actor-critic architecture
- **`source/whole_body_tracking/whole_body_tracking/utils/exporter.py`** — ONNX policy exporter
- **`source/whole_body_tracking/whole_body_tracking/robots/`** — Robot-specific settings
- **`scripts/`** — Utility scripts:
  - `csv_to_npz.py` — Single-file motion conversion
  - `csv_to_npz_inter_batch.py` — Batch motion conversion with start/end interpolation
  - `replay_npz.py` — Motion replay in Isaac Sim
  - `rsl_rl/train.py` — Policy training
  - `rsl_rl/play.py` — Policy playback & ONNX export
  - `rsl_rl/eval_metrics_folder.py` — Batch evaluation on a motion folder
