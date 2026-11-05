---
license: mit
task_categories:
- robotics
- reinforcement-learning
tags:
- humanoid-robot
- motion-capture
- unitree-g1
- retargeting
- locomotion
- dance
size_categories:
- n<1K
pretty_name: Unitree G1 Retargeted Motion Dataset
---

# Unitree G1 Retargeted Motion Dataset

A comprehensive collection of human motion data retargeted to the Unitree G1 humanoid robot. This dataset contains 174 motion sequences spanning diverse categories including locomotion, dance, sports, and expressive movements.

## Dataset Overview

This dataset provides motion data in a unified format suitable for training humanoid robot control policies. All motions have been retargeted from human motion capture data (SMPL format) to the Unitree G1 robot's kinematic structure using the [Mink retargeting pipeline](https://github.com/kevinzakka/mink).

**Total Files**: 174 motion sequences
**Robot**: Unitree G1 (23 DOF)
**Frame Rate**: 30 FPS
**Format**: Python pickle (.pkl)

## Subsets

### 1. ACCAD_retargeted (113 files)
Motion capture data from the CMU ACCAD dataset, covering a wide range of basic movements:
- **Locomotion**: Walking, running, crawling, lying down
- **Transitions**: Stand-to-sit, crouch-to-run, lie-to-crouch
- **Actions**: Picking up boxes, lifting, looking around
- **Dance**: Cartwheels, gestures

**Source**: [ACCAD - Carnegie Mellon University](http://mocap.cs.cmu.edu/)

### 2. LAFAN1_retargeted (40 files)
High-quality motion sequences from the LAFAN1 dataset:
- **Locomotion**: Multiple walking styles (walk1-4), running (run1-2), sprinting
- **Dance**: Contemporary dance sequences (dance1-2, 5 subjects each)
- **Dynamic Movements**: Jumps, falls and get-ups, fighting moves
- **Sports**: Fight and sports combinations

**Source**: [LAFAN1 Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)

### 3. dance_db_retargeted (13 files)
Curated selection of expressive dance and emotional movements:

**Dance Styles**:
- Latin: Bachata, Salsa, Reggaeton
- Street: Hip-hop, R&B
- Traditional: Zeibekiko, Greek folk dances
- Other: Capoeira, Flamenco, Zumba

**Emotional Expressions**:
- Positive: Happy, Excited, Relaxed
- Negative: Sad, Angry, Afraid

**Source**: DanceDB - Dance Motion Capture Database

### 4. kungfu_retargeted (8 files)
Martial arts and expressive poses:
- Bruce Lee pose
- Punches (Hooks, Horse stance)
- Kicks (Roundhouse, Side kick)
- Charleston dance

## Data Format

Each `.pkl` file contains a nested dictionary structure:

```python
{
    'motion_name.npz': {
        'root_trans_offset': np.ndarray,  # (T, 3) - Root position in world frame
        'root_rot': np.ndarray,           # (T, 4) - Root orientation (quaternion)
        'dof': np.ndarray,                # (T, 23) - Joint angles for 23 DOF
        'pose_aa': np.ndarray,            # (T, 27, 3) - Axis-angle representation
        'contact_mask': np.ndarray,       # (T, 2) - Foot contact labels [left, right]
        'fps': int,                       # Frame rate (30)
    }
}
```

Where `T` is the number of frames in the sequence.

### Joint Configuration (23 DOF)

The Unitree G1 robot has 23 degrees of freedom:
- **Torso**: 3 DOF (waist pitch, roll, yaw)
- **Arms**: 8 DOF per arm (shoulder, elbow, wrist)
- **Legs**: 6 DOF per leg (hip, knee, ankle)

## Usage

### Loading Data

```python
import pickle
import numpy as np

# Load a motion file
with open('ACCAD_retargeted/A1_-_Stand_stageii.pkl', 'rb') as f:
    data = pickle.load(f)

# Access motion data
motion_key = list(data.keys())[0]
motion = data[motion_key]

root_pos = motion['root_trans_offset']      # (T, 3)
root_rot = motion['root_rot']               # (T, 4)
joint_angles = motion['dof']                # (T, 23)
contacts = motion['contact_mask']           # (T, 2)

print(f"Sequence length: {len(root_pos)} frames")
print(f"Duration: {len(root_pos) / 30:.2f} seconds")
```

### Visualization

Requires the HumanoidPnnV2 codebase for visualization:

```bash
python humanoid-task-cluster/scripts/visualize_smpl_motion.py \
    --motion_file example/motion_data/ACCAD_retargeted/C3_-_Run_stageii.pkl
```

## Processing Pipeline

The motions were processed through the following pipeline:

1. **Source Data**: Human motion capture in SMPL/SMPL-X format
2. **Retargeting**: Mink inverse kinematics solver
3. **Contact Detection**: Physics-based foot contact detection
4. **Validation**: Manual inspection and filtering

## Citation

If you use this dataset, please cite:

```bibtex
@misc{g1_retargeted_motions_2025,
  title={Unitree G1 Retargeted Motion Dataset},
  author={Your Name},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/your-username/g1-retargeted-motions}}
}
```

### Original Data Sources

**ACCAD**:
```bibtex
@misc{accad_mocap,
  title={Advanced Computing Center for the Arts and Design Motion Capture Database},
  author={ACCAD},
  institution={The Ohio State University},
  url={http://accad.osu.edu/research/mocap/mocap_data.htm}
}
```

**LAFAN1**:
```bibtex
@article{harvey2020lafan1,
  title={Robust motion in-betweening},
  author={Harvey, F{\'e}lix G and Yurick, Mike and Nowrouzezahrai, Derek and Pal, Christopher},
  journal={ACM Transactions on Graphics (TOG)},
  volume={39},
  number={4},
  pages={60--1},
  year={2020}
}
```

## License

This dataset is provided for research purposes. Please respect the original licenses of the source datasets:
- ACCAD: [License](http://accad.osu.edu/research/mocap/mocap_usage.htm)
- LAFAN1: [License](https://github.com/ubisoft/ubisoft-laforge-animation-dataset#license)

## Related Resources

- **Retargeting Code**: [MaskedMimic](https://github.com/NVlabs/ProtoMotions)
- **Robot**: [Unitree G1](https://www.unitree.com/g1)
- **Training Framework**: [HumanoidPnnV2](https://github.com/your-repo)

## Updates

- **2025-12**: Initial release with 174 motion sequences
