---
license: apache-2.0
task_categories:
- robotics
- reinforcement-learning
tags:
- robotics
- humanoid-robot
- martial-arts
- kungfu
- motion-data
- robot-control
pretty_name: Kungfu Athlete Bot Dataset For Humanoid Robots
size_categories:
- n<1K
---

# KungfuAthleteBot
## Dataset Overview

The dataset originates from athletes’ **daily martial arts training videos**, totaling **197 video clips**.
Each clip may consist of multiple merged segments. We apply **automatic temporal segmentation**, resulting in **1,726 sub-clips**, ensuring that most segments avoid abrupt transitions that could introduce excessive motion discontinuities.

All sub-clips are processed using **GVHMR** for motion capture, followed by **GMR-based reorientation**.
After filtering and post-processing, the final dataset contains **848 motion samples**, primarily reflecting routine training activities.

Due to substantial noise in the source videos, the dataset has undergone multiple rounds of manual screening and meticulous per-sample refinement. While minor imperfections may still remain in certain samples, we ensure that the vast majority satisfy the requirements for reliable motion tracking.

> **Project Status: Active Development with Ready Ground Subset**
> Model training is currently under active development.
> The **Ground** subset of the dataset is largely complete and ready for training.
> The **Jump** subset still has minor imperfections due to video source limitations. Most samples have been carefully screened, though training performance may vary.
> Your feedback and suggestions are greatly appreciated. 
## Dataset Category Distribution

| Category       | Count  | Example Subcategories                                       |
| -------------- | ------ | ----------------------------------------------------------- |
| Daily Training | 624    | Training(624)                                               |
| Saber / Sword  | 111    | Saber(50), Sword(37), Southern Saber(15), Tai Chi Sword(9)  |
| Fist           | 98     | Long Fist(69), Tai Chi Fist(21), Southern Fist(8)           |
| Staff          | 90     | Staff Technique(90)                                         |
| Skills         | 69     | Side Flip(35), Air Spin(18), Lotus Swing(13), Backflip(3)   |


| Category | Total | Subcategory          | Count |
| -------- | ----- | --------------------| ----- |
| Ground   | 822   | Daily Training       | 550   |
|          |       | Fist                | 93    |
|          |       | Saber / Sword        | 99    |
|          |       | Staff               | 80    |
| Jump     | 170   | Daily Training       | 74    |
|          |       | Fist                | 5     |
|          |       | Saber / Sword        | 12    |
|          |       | Skills              | 69    |
|          |       | - Side Flip          | 35    |
|          |       | - Air Spin           | 18    |
|          |       | - Lotus Swing        | 13    |
|          |       | - Backflip           | 3     |
|          |       | - Other Skills       | 0     |
|          |       | Staff               | 10    |

### Notes

* The dataset consists of **992 motion samples** in total, divided into **Ground (822)** and **Jump (170)** categories.
* **Daily Training** accounts for the largest proportion, with **624 samples** across both ground and jump motions, mainly reflecting basic routine movements.
* Ground motions include **Fist (93), Saber / Sword (99)**, and **Staff (80)**; jump motions include a small number of **Fist (5), Saber / Sword (12)**, and **Staff (10)** techniques.
* **Skill-based movements** only exist in jump motions, totaling **69 samples**, focusing on high-difficulty acrobatics such as **Side Flip (35), Air Spin (18), Lotus Swing (13)**, and **Backflip (3)**.
* Weapon-related motions only contain overall movement sequences and do not include detailed weapon or hand operations (to be updated in future versions).
---

## Motion Statistics Comparison

All metrics are averaged over the entire dataset.

| Dataset                    | Body Lin Vel(w) | Body Ang Vel(w) | Frames Range |
| -------------------------- | --------------- | --------------- | ------------ |
| AMASS                      | 0.188±0.165     | 0.510±0.309     | [2,8788]     |
| PHUMA                      | 0.193±0.130     | 0.585±0.368     | [49,222]     |
| LAFAN1                     | 0.790±0.251     | 1.962±0.553     | [3065,9171]  |
| **KungFuAthlete (Ground)** | 0.496±0.324     | 1.537±0.959     | [24,15792]   |
| **KungFuAthlete (Jump)**   | 0.948±0.473     | 2.832±1.431     | [25,2842]    |

---

## Ground vs. Jump Subsets

We divide the dataset based on the presence of jumping motions:

* **KungFuAthlete (Ground)**
  Contains non-jumping actions, emphasizing:

  * Continuous ground-based power generation
  * Rapid body rotations
  * Weapon manipulation and stance transitions

* **KungFuAthlete (Jump)**
  Includes high-dynamic aerial motions such as:

  * Somersaults
  * Cartwheels
  * Other acrobatic jumps

### Key Observations

* The **Jump subset** exhibits the **highest joint velocity, body linear velocity, and angular velocity** among all compared datasets.
* The **Ground subset**, while excluding jumps, still shows significantly higher dynamics than natural motion datasets (e.g., LAFAN1, AMASS).
* Compared to PHUMA and AMASS, which focus on daily activities and walking motions, **KungFuAthlete demonstrates stronger non-stationarity, larger motion amplitudes, and more challenging transient dynamics**, even at comparable or higher frame rates.

### ⚠️ Safety Warning

> The *Jump* subset is intended for studying the upper limits of humanoid motion. Direct training or deployment on real robots without strict safety constraints may lead to serious hardware damage.

## Demo

<table>
<tr>
<td align="center" width="50%">

![](./docs/example_gvhmr_278.gif)  
**278 GVMR SMPL**

</td>
<td align="center" width="50%">

![](./docs/example_g1_after_278.gif)  
**278 GMR, Height-Adjusted**

</td>
</tr>
</table>

## Pipeline

KungfuAthlete's data pipeline: 

```
[Video] → Cut by scene → GVHMR → [GVHMR-Pred (smplh)] → GMR → [Robot Motion (qpos)] → Artificial Selection → Height-Adjusted → [KungfuAthlete]
```

### Supported Robots

| Robot | ID | DOF |
|-------|-----|-----|
| Unitree G1 | `unitree_g1` | 29 |

See [GMR README](https://github.com/YanjieZe/GMR) for other list


## Data Format

### GVHMR pred

```python
# gvhmr_pred.pt
{
  "smpl_params_global":
  {
    "body_pose": torch.Tensor,    # (N, 63)
    "betas": torch.Tensor,    # (N, 10)
    "global_orient": torch.Tensor,    # (N, 3)
    "transl": torch.Tensor,    # (N, 3)
  }
  "smpl_params_incam":
  ...
}
```

If you intend to perform motion retargeting to other robotic platforms, or to employ alternative retargeting methodologies, please use the SMPL-H files provided in the  `collection_gvhmr` directory.

### GMR qpos (`org_smoothed/`)

```python
# robot_qpos.npz
{
  "fps": array(30),
  "qpos": np.ndarray,    # (N, 36) 36 = 3(position xyz) + 4(quaternion wxyz) + 29(DOF)
}
```


### For BeyondMimic (`org_smoothed_mj/`)

All motion files under `org_smoothed_mj/` are stored in NumPy `.npz` format and contain time-sequential kinematic data for Unitree G1 training. Each file represents a single motion clip sampled at `fps` Hz and includes joint-level states and rigid-body states expressed in the **world coordinate frame** (quaternion format: xyzw; units: meters, radians, m/s, rad/s).

```python
motion_clip = {
    "fps": np.array([50]),               # Frame rate (Hz)
    "joint_pos": np.ndarray,             # Shape (T, 36), joint DoF positions (rad)
    "joint_vel": np.ndarray,             # Shape (T, 35), joint angular velocities (rad/s)
    "body_pos_w": np.ndarray,            # Shape (T, 53, 3), rigid body positions in world frame (m)
    "body_quat_w": np.ndarray,           # Shape (T, 53, 4), rigid body orientations in world frame (xyzw)
    "body_lin_vel_w": np.ndarray,        # Shape (T, 53, 3), rigid body linear velocities (m/s)
    "body_ang_vel_w": np.ndarray,        # Shape (T, 53, 3), rigid body angular velocities (rad/s)
    "joint_names": np.ndarray,           # Shape (29,), controllable joint names (ordered)
    "body_names": np.ndarray,            # Shape (53,), rigid body names (ordered, includes head_link)
}
```

Here, `T` denotes the number of frames in the motion clip.