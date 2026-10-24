#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument("config_file", default_value=[PathJoinSubstitution([FindPackageShare("textop_ctrl"), "config", "g1_29dof.yaml"])], description="Path to the configuration file")

    onnx_model_arg = DeclareLaunchArgument("onnx_model", default_value=[PathJoinSubstitution([FindPackageShare("textop_ctrl"), "models", "policy.onnx"])], description="Path to the ONNX model file")

    motion_file_arg = DeclareLaunchArgument("motion_file", default_value=[PathJoinSubstitution([FindPackageShare("textop_ctrl"), "models", "motion.npz"])], description="Path to the motion file")

    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="false", description="Use simulation time")

    # Create the node
    textop_controller_node = Node(
        package="textop_ctrl",
        executable="textop_onnx_controller",
        name="textop_onnx_controller",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "config_path": LaunchConfiguration("config_file"),
                "onnx_model_path": LaunchConfiguration("onnx_model"),
                "motion_path": LaunchConfiguration("motion_file"),
            }
        ],
    )

    # Log info
    log_info = LogInfo(msg=[TextSubstitution(text="Starting TextOp ONNX Controller with config: "), LaunchConfiguration("config_file")])

    return LaunchDescription(
        [
            config_file_arg,
            onnx_model_arg,
            motion_file_arg,
            use_sim_time_arg,
            log_info,
            textop_controller_node,
        ]
    )
