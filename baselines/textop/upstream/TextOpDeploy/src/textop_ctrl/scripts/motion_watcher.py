#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from textop_ctrl.msg import MotionBlock
from std_msgs.msg import Float32MultiArray
import numpy as np
import time
import threading
import os
from pathlib import Path

np.set_printoptions(precision=6, suppress=True, linewidth=200)


class MotionWatcher(Node):

    def __init__(self, need_save=True):
        super().__init__('motion_watcher')

        # Visualization control
        self.need_save = need_save
        self.is_visualizing = True
        self.dt = 1.0 / 50.0  # 50Hz
        self.current_frame_index = 0
        self.toggle_callback()

        # Mujoco visualization
        self.mjc = self.load_mujoco()

        # Subscribers
        self.motion_sub = self.create_subscription(MotionBlock, '/dar/motion',
                                                   self.motion_block_callback,
                                                   10)

        self.toggle_sub = self.create_subscription(Time, '/dar/toggle',
                                                   self.toggle_callback, 10)

        # Visualization timer (50Hz)
        self.viz_timer = self.create_timer(self.dt, self.visualization_loop)

        self.get_logger().info("Motion Watcher initialized")

    def load_mujoco(self):
        """Initialize MuJoCo visualization"""
        try:
            import mujoco
            import mujoco.viewer

            # Use relative path from current working directory
            print(os.getcwd())
            humanoid_xml = "./src/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
            if not os.path.exists(humanoid_xml):
                self.get_logger().error(
                    f"Could not find g1_29dof.xml in any expected location")
                return None

            mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
            mj_data = mujoco.MjData(mj_model)
            mj_model.opt.timestep = 1 / 50  # 50Hz visualization

            viewer = mujoco.viewer.launch_passive(mj_model,
                                                  mj_data,
                                                  show_left_ui=False,
                                                  show_right_ui=False)
            viewer.cam.lookat[:] = np.array([0, 0, 0.7])
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = -130
            viewer.cam.elevation = -20

            self.get_logger().info(f"MuJoCo loaded from: {humanoid_xml}")
            return (mujoco, mj_model, mj_data, viewer)

        except ImportError:
            self.get_logger().error(
                "MuJoCo not available, visualization disabled")
            return None
        except Exception as e:
            self.get_logger().error(f"Failed to load MuJoCo: {e}")
            return None

    def step_mujoco(self, mjc, q, pos, quat):
        """Step MuJoCo simulation with given joint positions and root pose"""
        if mjc is None:
            return

        mujoco, mj_model, mj_data, viewer = mjc
        try:
            # Set root position and orientation
            mj_data.qpos[:3] = pos
            mj_data.qpos[3:7] = quat  # WXYZ quaternion

            # Set joint positions (first 29 joints)
            if len(q) >= 29:
                mj_data.qpos[7:36] = q[:29]
            else:
                mj_data.qpos[7:7 + len(q)] = q

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

        except Exception as e:
            self.get_logger().error(f"Error in MuJoCo step: {e}")

    def extract_array_data(self, float32_array):
        """Extract data from Float32MultiArray message"""
        return list(float32_array.data)

    def motion_block_callback(self, msg):
        """Callback for MotionBlock messages"""
        # Extract array data
        joint_positions = self.extract_array_data(msg.joint_positions)
        joint_velocities = self.extract_array_data(msg.joint_velocities)
        anchor_body_ori = self.extract_array_data(msg.anchor_body_ori)
        anchor_body_pos = self.extract_array_data(msg.anchor_body_pos)

        # Extract dimensions
        T_block = 0
        Nq = 0

        if (len(msg.joint_positions.layout.dim) >= 2):
            T_block = msg.joint_positions.layout.dim[0].size
            Nq = msg.joint_positions.layout.dim[1].size
            self.motion_sequence['num_joints'] = Nq
        else:
            # Fallback: infer from data size
            if len(joint_positions
                   ) > 0 and self.motion_sequence['num_joints'] > 0:
                T_block = len(
                    joint_positions) // self.motion_sequence['num_joints']
                Nq = self.motion_sequence['num_joints']

        if T_block <= 0 or Nq <= 0:
            self.get_logger().warn(
                f"Invalid block dimensions (T={T_block}, Nq={Nq})")
            return

        block_index = msg.index
        required_length = block_index + T_block
        if required_length <= self.motion_sequence['total_length']:
            self.current_frame_index = block_index

        # Ensure sequence capacity
        self.ensure_sequence_capacity(required_length)

        # Update motion sequence
        for t in range(T_block):
            seq_idx = block_index + t

            # Joint positions [T, Nq]
            if seq_idx < len(self.motion_sequence['joint_positions']):
                for j in range(Nq):
                    self.motion_sequence['joint_positions'][seq_idx][
                        j] = joint_positions[t * Nq + j]

            # Joint velocities [T, Nq]
            if seq_idx < len(self.motion_sequence['joint_velocities']):
                for j in range(Nq):
                    self.motion_sequence['joint_velocities'][seq_idx][
                        j] = joint_velocities[t * Nq + j]

            # Anchor body orientation [T, 4]
            if seq_idx < len(self.motion_sequence['anchor_body_ori']):
                for i in range(4):
                    self.motion_sequence['anchor_body_ori'][seq_idx][
                        i] = anchor_body_ori[t * 4 + i]

            # Anchor body position [T, 3]
            if seq_idx < len(self.motion_sequence['anchor_body_pos']):
                for i in range(3):
                    self.motion_sequence['anchor_body_pos'][seq_idx][
                        i] = anchor_body_pos[t * 3 + i]

        # Update total length
        self.motion_sequence['total_length'] = max(
            self.motion_sequence['total_length'], required_length)

        self.get_logger().info(
            f"Received MotionBlock #{block_index}: T={T_block}, total_length={self.motion_sequence['total_length']}"
        )

        # Save data after each block
        if self.need_save:
            self.save_motion_to_npz()

    def ensure_sequence_capacity(self, required_length):
        """Ensure motion sequence has enough capacity"""
        current_length = len(self.motion_sequence['joint_positions'])
        if required_length > current_length:
            # Extend sequences
            for i in range(current_length, required_length):
                self.motion_sequence['joint_positions'].append(
                    [0.0] * self.motion_sequence['num_joints'])
                self.motion_sequence['joint_velocities'].append(
                    [0.0] * self.motion_sequence['num_joints'])
                self.motion_sequence['anchor_body_ori'].append(
                    [0.0, 0.0, 0.0, 1.0])  # Identity quaternion
                self.motion_sequence['anchor_body_pos'].append([0.0, 0.0, 0.0])

    def toggle_callback(self, msg=None):
        """Reset buffer"""
        self.current_frame_index = 0
        # Motion data storage
        self.motion_sequence = {
            'joint_positions': [],  # [Total_T, Nq]
            'joint_velocities': [],  # [Total_T, Nq]
            'anchor_body_ori': [],  # [Total_T, 4]
            'anchor_body_pos': [],  # [Total_T, 3]
            'total_length': 0,
            'num_joints': 29,
            'fps': 50
        }

    isaaclab_to_mujoco_reindex = [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23,
        25, 27, 12, 16, 20, 22, 24, 26, 28
    ]

    def visualization_loop(self):
        """50Hz visualization loop"""
        if not self.is_visualizing:
            return
        if self.mjc is None:
            raise RuntimeError("MuJoCo died")

        if self.motion_sequence['total_length'] == 0:
            return

        # Get current frame data
        frame_idx = np.clip(self.current_frame_index, 0,
                            self.motion_sequence['total_length'] - 1)

        time0 = self.get_clock().now()
        if frame_idx < len(self.motion_sequence['joint_positions']):
            # Get joint positions
            q = np.array(self.motion_sequence['joint_positions'][frame_idx])
            q = q[self.isaaclab_to_mujoco_reindex]

            # Get root position and orientation
            pos = np.array(self.motion_sequence['anchor_body_pos'][frame_idx])
            quat = np.array(self.motion_sequence['anchor_body_ori'][frame_idx])

            # Step MuJoCo visualization
            self.step_mujoco(self.mjc, q, pos, quat)

            # Advance frame
            time1 = self.get_clock().now()
            elapsed_sec = (time1 - time0).nanoseconds / 1e9
            current_counter = np.max((1, np.floor(elapsed_sec / self.dt)))
            self.current_frame_index += int(current_counter)
            # self.current_frame_index += 1
            self.get_logger().info(
                f"Frame {self.current_frame_index}/{self.motion_sequence['total_length']}"
            )

    def save_motion_to_npz(self):
        """Save current motion sequence to NPZ file"""
        if self.motion_sequence['total_length'] == 0:
            return

        try:
            # Create output directory
            output_dir = Path("log_motion")
            output_dir.mkdir(exist_ok=True)

            # Fixed filename (overwrite each time)
            filename = output_dir / "motion_watcher_data.npz"

            # Prepare data arrays
            T = self.motion_sequence['total_length']
            Nq = self.motion_sequence['num_joints']

            # Convert to numpy arrays
            joint_pos = np.array(
                self.motion_sequence['joint_positions'][:T])  # [T, Nq]
            joint_vel = np.array(
                self.motion_sequence['joint_velocities'][:T])  # [T, Nq]
            anchor_ori = np.array(
                self.motion_sequence['anchor_body_ori'][:T])  # [T, 4]
            anchor_pos = np.array(
                self.motion_sequence['anchor_body_pos'][:T])  # [T, 3]

            # Reshape for compatibility with expected format
            body_pos_w = anchor_pos.reshape(T, 1, 3)  # [T, 1, 3]
            body_quat_w = anchor_ori.reshape(T, 1, 4)  # [T, 1, 4]

            # Save to NPZ
            np.savez(filename,
                     joint_pos=joint_pos,
                     joint_vel=joint_vel,
                     body_pos_w=body_pos_w,
                     body_quat_w=body_quat_w,
                     fps=np.array([self.motion_sequence['fps']]))

            self.get_logger().debug(
                f"Saved motion data: T={T}, Nq={Nq} to {filename}")

        except Exception as e:
            self.get_logger().error(f"Failed to save motion data: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        motion_watcher = MotionWatcher(need_save=True)
        rclpy.spin(motion_watcher)
    except KeyboardInterrupt:
        print("Shutting down Motion Watcher...")
    finally:
        if 'motion_watcher' in locals():
            motion_watcher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
