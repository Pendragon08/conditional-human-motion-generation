#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    robot_arg = DeclareLaunchArgument("robot",
                                      default_value="g1",
                                      description="Robot type (g1 or go2)")

    interface_arg = DeclareLaunchArgument("interface",
                                          default_value="eno1",
                                          description="Network interface")

    domain_id_arg = DeclareLaunchArgument("domain_id",
                                          default_value="0",
                                          description="DDS domain ID")

    # Execute the MuJoCo simulator
    mujoco_simulate = ExecuteProcess(cmd=[
        "unitree_mujoco_ros", "-r",
        LaunchConfiguration("robot"), "-n",
        LaunchConfiguration("interface"), "-i",
        LaunchConfiguration("domain_id")
    ],
                                     output="screen",
                                     name="mujoco_simulate")

    return LaunchDescription([
        robot_arg,
        interface_arg,
        domain_id_arg,
        mujoco_simulate,
    ])
