#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from builtin_interfaces.msg import Time
from textop_ctrl.msg import MotionBlock
import numpy as np
import argparse
import os


def expand_dof_23_to_29(v: np.ndarray) -> np.ndarray:
    """
    v: shape [T, 23]
    return: shape [T, 29]
    """
    T = v.shape[0]
    out = np.zeros((T, 29), dtype=v.dtype)

    out[:, :19] = v[:, :19]
    out[:, 22:26] = v[:, 19:23]

    return out


class NPZMotionPublisher(Node):

    def __init__(self,
                 npz_file_path,
                 mode='batch',
                 batch_size=8,
                 publish_rate=2.0):
        super().__init__('npz_motion_publisher')

        self.npz_file_path = npz_file_path
        self.mode = mode  # 'single' or 'batch'
        self.batch_size = batch_size
        self.publish_rate = publish_rate

        # Load NPZ data
        self.load_npz_data()

        # Create publisher for MotionBlock messages
        self.publisher = self.create_publisher(MotionBlock, '/dar/motion', 10)

        # Initialize publishing state
        self.current_index = 0
        self.total_published_blocks = 0
        self.is_finished = False

        # Create timer based on mode
        if self.mode == 'single':
            # Publish once immediately
            self.timer = self.create_timer(0.1, self.publish_single_block)
        else:
            # Publish in batches at specified rate
            timer_period = 1.0 / self.publish_rate  # Convert Hz to seconds
            self.timer = self.create_timer(timer_period,
                                           self.publish_batch_block)

        self.get_logger().info(f'NPZ Motion Publisher started')
        self.get_logger().info(f'  - File: {npz_file_path}')
        self.get_logger().info(f'  - Mode: {mode}')
        self.get_logger().info(
            f'  - Total motion length: {self.total_T} timesteps')
        self.get_logger().info(f'  - Number of joints: {self.num_joints}')
        self.get_logger().info(f'  - FPS: {self.fps}')
        if mode == 'batch':
            self.get_logger().info(f'  - Batch size: {batch_size}')
            self.get_logger().info(f'  - Publish rate: {publish_rate} Hz')

    def load_npz_data(self):
        """Load motion data from NPZ file"""
        if not os.path.exists(self.npz_file_path):
            self.get_logger().error(
                f'NPZ file not found: {self.npz_file_path}')
            raise FileNotFoundError(
                f'NPZ file not found: {self.npz_file_path}')

        try:
            data = np.load(self.npz_file_path)

            # Load joint positions [T, 29]
            if 'joint_pos' in data:
                self.joint_pos = data[
                    'joint_pos']  # Shape: [T, Nq], isaaclab order
                self.total_T, self.num_joints = self.joint_pos.shape
                if self.num_joints == 23:
                    raise NotImplementedError(
                        '23 dof motion means mujoco order, should never be used'
                    )
                    self.num_joints = 29
                    self.joint_pos = expand_dof_23_to_29(self.joint_pos)
            else:
                self.get_logger().error('joint_pos not found in NPZ file')
                raise KeyError('joint_pos not found in NPZ file')

            # Load joint velocities [T, 29]
            if 'joint_vel' in data:
                self.joint_vel = data['joint_vel']  # Shape: [T, Nq]
                if self.joint_vel.shape[1] == 23:
                    self.joint_vel = expand_dof_23_to_29(self.joint_vel)
            else:
                self.get_logger().warn('joint_vel not found, using zeros')
                self.joint_vel = np.zeros((self.total_T, self.num_joints))

            # Load body positions [T, N, 3] - we use only the first body (anchor body)
            if 'body_pos_w' in data:
                body_pos_w = data['body_pos_w']  # Shape: [T, N, 3]
                if len(body_pos_w.shape) == 3:
                    self.body_pos = body_pos_w[:,
                                               0, :]  # Take first body: [T, 3]
                else:
                    self.get_logger().error('body_pos_w has unexpected shape')
                    raise ValueError('body_pos_w has unexpected shape')
            else:
                self.get_logger().warn('body_pos_w not found, using zeros')
                self.body_pos = np.zeros((self.total_T, 3))

            # Load body orientations [T, N, 4] - we use only the first body (anchor body)
            if 'body_quat_w' in data:
                body_quat_w = data['body_quat_w']  # Shape: [T, N, 4]
                if len(body_quat_w.shape) == 3:
                    self.body_ori = body_quat_w[:,
                                                0, :]  # Take first body: [T, 4]
                else:
                    self.get_logger().error('body_quat_w has unexpected shape')
                    raise ValueError('body_quat_w has unexpected shape')
            else:
                self.get_logger().warn(
                    'body_quat_w not found, using identity quaternions')
                self.body_ori = np.zeros((self.total_T, 4))
                self.body_ori[:,
                              0] = 1.0  # w=1, x=y=z=0 for identity quaternion

            # Load FPS
            if 'fps' in data:
                fps_array = data['fps']
                if fps_array.shape == ():  # scalar
                    self.fps = int(fps_array)
                elif fps_array.shape == (1, ):  # single element array
                    self.fps = int(fps_array[0])
                else:
                    self.get_logger().warn(
                        'fps has unexpected shape, using default 30')
                    self.fps = 30
            else:
                self.get_logger().warn('fps not found, using default 30')
                self.fps = 30

            data.close()

            self.get_logger().info(f'Successfully loaded NPZ data:')
            self.get_logger().info(f'  - joint_pos: {self.joint_pos.shape}')
            self.get_logger().info(f'  - joint_vel: {self.joint_vel.shape}')
            self.get_logger().info(f'  - body_pos: {self.body_pos.shape}')
            self.get_logger().info(f'  - body_ori: {self.body_ori.shape}')
            self.get_logger().info(f'  - fps: {self.fps}')

        except Exception as e:
            self.get_logger().error(f'Error loading NPZ file: {str(e)}')
            raise

    def create_float32_multi_array(self, data, dimensions_info):
        """
        Helper function to create Float32MultiArray with proper dimension info
        
        Args:
            data: numpy array or list of data
            dimensions_info: list of tuples (label, size, stride)
        """
        msg = Float32MultiArray()

        # Set data
        if isinstance(data, np.ndarray):
            msg.data = data.flatten().astype(np.float32).tolist()
        else:
            msg.data = [float(x) for x in data]

        # Set dimensions
        msg.layout.dim = []
        for label, size, stride in dimensions_info:
            dim = MultiArrayDimension()
            dim.label = label
            dim.size = size
            dim.stride = stride
            msg.layout.dim.append(dim)

        msg.layout.data_offset = 0

        return msg

    def publish_single_block(self):
        """Publish all data as a single block"""
        if self.is_finished:
            return

        msg = MotionBlock()

        # Set index to 0 (start from beginning)
        msg.index = 0

        # Set current timestamp
        current_time = self.get_clock().now().to_msg()
        msg.timestamp = current_time

        T = self.total_T  # Use full length
        Nq = self.num_joints

        # ===== Joint Positions [T, Nq] =====
        msg.joint_positions = self.create_float32_multi_array(
            self.joint_pos, [("time_steps", T, T * Nq), ("joints", Nq, Nq)])

        # ===== Joint Velocities [T, Nq] =====
        msg.joint_velocities = self.create_float32_multi_array(
            self.joint_vel, [("time_steps", T, T * Nq), ("joints", Nq, Nq)])

        # ===== Anchor Body Orientation [T, 4] =====
        msg.anchor_body_ori = self.create_float32_multi_array(
            self.body_ori, [("time_steps", T, T * 4), ("quaternion", 4, 4)])

        # ===== Anchor Body Position [T, 3] =====
        msg.anchor_body_pos = self.create_float32_multi_array(
            self.body_pos, [("time_steps", T, T * 3), ("position", 3, 3)])

        # Publish the message
        self.publisher.publish(msg)

        self.get_logger().info(f'Published SINGLE MotionBlock with full data:')
        self.get_logger().info(f'  - Index: {msg.index}')
        self.get_logger().info(f'  - Total timesteps: {T}')
        self.get_logger().info(f'  - Joint positions: [{T}, {Nq}]')
        self.get_logger().info(f'  - Joint velocities: [{T}, {Nq}]')
        self.get_logger().info(f'  - Anchor body ori: [{T}, 4]')
        self.get_logger().info(f'  - Anchor body pos: [{T}, 3]')

        self.is_finished = True
        self.timer.cancel()  # Stop the timer

        self.get_logger().info('Single block publishing completed!')
        raise Exception('Node Action Completed!')

    def publish_batch_block(self):
        """Publish data in batches"""
        if self.is_finished:
            return

        # Calculate the current batch
        start_idx = self.current_index
        end_idx = min(start_idx + self.batch_size, self.total_T)
        actual_batch_size = end_idx - start_idx

        if actual_batch_size <= 0:
            self.get_logger().info('All batches published!')
            self.is_finished = True
            self.timer.cancel()
            raise Exception('Node Action Completed!')
            return

        msg = MotionBlock()

        # Set index (starting position in the continuous sequence)
        msg.index = start_idx

        # Set current timestamp
        current_time = self.get_clock().now().to_msg()
        msg.timestamp = current_time

        T = actual_batch_size
        Nq = self.num_joints

        # Extract batch data
        joint_pos_batch = self.joint_pos[start_idx:end_idx, :]  # [T, Nq]
        joint_vel_batch = self.joint_vel[start_idx:end_idx, :]  # [T, Nq]
        body_ori_batch = self.body_ori[start_idx:end_idx, :]  # [T, 4]
        body_pos_batch = self.body_pos[start_idx:end_idx, :]  # [T, 3]

        # ===== Joint Positions [T, Nq] =====
        msg.joint_positions = self.create_float32_multi_array(
            joint_pos_batch, [("time_steps", T, T * Nq), ("joints", Nq, Nq)])

        # ===== Joint Velocities [T, Nq] =====
        msg.joint_velocities = self.create_float32_multi_array(
            joint_vel_batch, [("time_steps", T, T * Nq), ("joints", Nq, Nq)])

        # ===== Anchor Body Orientation [T, 4] =====
        msg.anchor_body_ori = self.create_float32_multi_array(
            body_ori_batch, [("time_steps", T, T * 4), ("quaternion", 4, 4)])

        # ===== Anchor Body Position [T, 3] =====
        msg.anchor_body_pos = self.create_float32_multi_array(
            body_pos_batch, [("time_steps", T, T * 3), ("position", 3, 3)])

        # Publish the message
        self.publisher.publish(msg)

        self.total_published_blocks += 1
        remaining_timesteps = self.total_T - end_idx

        self.get_logger().info(
            f'Published MotionBlock #{self.total_published_blocks}:')
        self.get_logger().info(
            f'  - Index range: {start_idx}-{end_idx-1} (T={T})')
        self.get_logger().info(
            f'  - Remaining timesteps: {remaining_timesteps}')

        # Update current index for next batch
        self.current_index = end_idx


def main(args=None):
    rclpy.init(args=args)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Publish NPZ motion data as MotionBlock messages')
    parser.add_argument('npz_file', help='Path to NPZ motion file')
    parser.add_argument(
        '--mode',
        choices=['single', 'batch'],
        default='batch',
        help='Publishing mode: single (all at once) or batch (in chunks)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Number of timesteps per batch (only for batch mode)')
    parser.add_argument('--rate',
                        type=float,
                        default=6.25,
                        help='Publishing rate in Hz (only for batch mode)')

    # Parse known args to handle ROS2 arguments
    parsed_args, unknown = parser.parse_known_args()

    try:
        npz_motion_publisher = NPZMotionPublisher(
            npz_file_path=parsed_args.npz_file,
            mode=parsed_args.mode,
            batch_size=parsed_args.batch_size,
            publish_rate=parsed_args.rate)

        rclpy.spin(npz_motion_publisher)

    except Exception as e:
        print(f'Error: {e}')
        return 1
    finally:
        if 'npz_motion_publisher' in locals():
            npz_motion_publisher.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    exit(main())
