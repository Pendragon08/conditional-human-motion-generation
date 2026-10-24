# TEXTOP ONNX Controller

This package provides a C++ ROS2 implementation for deploying TextOp policy on 29DOF G1 robot.

## Dependencies

- ROS2
- ONNX Runtime
- unitree ros2 
- unitree mujoco (Modified by Weiji)
- cnpy (for NPZ file loading)
- yaml-cpp

## Configuration

The controller uses a YAML configuration file (`config/g1_29dof.yaml`) with the following key parameters:

- `control_dt`: Control loop period (default: 0.02s = 50Hz)
- `onnx_path`: Path to the ONNX model file
- `lowcmd_topic`: ROS2 topic for low-level commands
- `lowstate_topic`: ROS2 topic for robot state
- Joint mappings and control gains for 29DOF
- Scaling factors for observations and actions

## Architecture

### Main Components

1. **ONNXPolicy**: Handles ONNX model loading and inference
2. **MotionBlockSubscriber**: Loads and manages motion data from ROS2 topic
3. **ObservationComputer**: Computes observations matching Python implementation
4. **TextOpOnnxController**: Main ROS2 node with control loop
5. **Helper Classes**: Command helpers, rotation utilities, remote controller

### Control Loop

1. Receive robot state from ROS2 topics
2. Load motion data from NPZ file or Diffusion Model
3. Process sensor data (IMU, joint states)
4. Create observation vector matching training setting
5. Run ONNX inference to get actions
6. Transform actions to motor commands using 29DOF mapping
7. Send commands via ROS2
8. Safety checks and timing validation

### Motion Data Format

The motion loader expects NPZ files with the following arrays:
- `joint_pos`: Joint positions [T, 29]
- `joint_vel`: Joint velocities [T, 29]
- `body_pos_w`: Body positions in world frame [T, N, 3]
- `body_quat_w`: Body orientations in world frame [T, N, 4]
- `body_ang_vel_w`: Body angular velocities [T, N, 3]
- `fps`: Frame rate (integer)
