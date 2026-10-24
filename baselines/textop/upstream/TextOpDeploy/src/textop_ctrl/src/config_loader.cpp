#include "textop_ctrl/config_loader.hpp"

#include <fstream>
#include <iostream>

Config::Config(const std::string& config_path)
{
    try
    {
        YAML::Node config = YAML::LoadFile(config_path);

        // Control parameters
        if (config["control_dt"])
        {
            control_dt = config["control_dt"].as<double>();
        }
        if (config["msg_type"])
        {
            msg_type = config["msg_type"].as<std::string>();
        }
        if (config["imu_type"])
        {
            imu_type = config["imu_type"].as<std::string>();
        }

        // Topics
        if (config["lowcmd_topic"])
        {
            lowcmd_topic = config["lowcmd_topic"].as<std::string>();
        }
        if (config["lowstate_topic"])
        {
            lowstate_topic = config["lowstate_topic"].as<std::string>();
        }

        // Model paths
        if (config["policy_path"])
        {
            policy_path = config["policy_path"].as<std::string>();
        }
        if (config["onnx_path"])
        {
            onnx_path = config["onnx_path"].as<std::string>();
        }
        if (config["motion_path"])
        {
            motion_path = config["motion_path"].as<std::string>();
        }

        // Control gains for 29DOF
        if (config["kps"])
        {
            kps = config["kps"].as<std::vector<double>>();
        }
        else
        {
            exit(1);
            // Default stiffness values based on Python script
            kps = {
                40.17923847137318,  // left_hip_pitch_joint
                99.09842777666113,  // left_hip_roll_joint
                40.17923847137318,  // left_hip_yaw_joint
                99.09842777666113,  // left_knee_joint
                28.50124619574858,  // left_ankle_pitch_joint
                28.50124619574858,  // left_ankle_roll_joint
                40.17923847137318,  // right_hip_pitch_joint
                99.09842777666113,  // right_hip_roll_joint
                40.17923847137318,  // right_hip_yaw_joint
                99.09842777666113,  // right_knee_joint
                28.50124619574858,  // right_ankle_pitch_joint
                28.50124619574858,  // right_ankle_roll_joint
                40.17923847137318,  // waist_yaw_joint
                28.50124619574858,  // waist_roll_joint
                28.50124619574858,  // waist_pitch_joint
                14.25062309787429,  // left_shoulder_pitch_joint
                14.25062309787429,  // left_shoulder_roll_joint
                14.25062309787429,  // left_shoulder_yaw_joint
                14.25062309787429,  // left_elbow_joint
                14.25062309787429,  // left_wrist_roll_joint
                16.77832748089279,  // left_wrist_pitch_joint
                16.77832748089279,  // left_wrist_yaw_joint
                14.25062309787429,  // right_shoulder_pitch_joint
                14.25062309787429,  // right_shoulder_roll_joint
                14.25062309787429,  // right_shoulder_yaw_joint
                14.25062309787429,  // right_elbow_joint
                14.25062309787429,  // right_wrist_roll_joint
                16.77832748089279,  // right_wrist_pitch_joint
                16.77832748089279   // right_wrist_yaw_joint
            };
        }

        if (config["kds"])
        {
            kds = config["kds"].as<std::vector<double>>();
        }
        else
        {
            exit(1);
            // Default damping values based on Python script
            kds = {
                2.5578897650279457,  // left_hip_pitch_joint
                6.3088018534966395,  // left_hip_roll_joint
                2.5578897650279457,  // left_hip_yaw_joint
                6.3088018534966395,  // left_knee_joint
                1.814445686584846,   // left_ankle_pitch_joint
                1.814445686584846,   // left_ankle_roll_joint
                2.5578897650279457,  // right_hip_pitch_joint
                6.3088018534966395,  // right_hip_roll_joint
                2.5578897650279457,  // right_hip_yaw_joint
                6.3088018534966395,  // right_knee_joint
                1.814445686584846,   // right_ankle_pitch_joint
                1.814445686584846,   // right_ankle_roll_joint
                2.5578897650279457,  // waist_yaw_joint
                1.814445686584846,   // waist_roll_joint
                1.814445686584846,   // waist_pitch_joint
                0.907222843292423,   // left_shoulder_pitch_joint
                0.907222843292423,   // left_shoulder_roll_joint
                0.907222843292423,   // left_shoulder_yaw_joint
                0.907222843292423,   // left_elbow_joint
                0.907222843292423,   // left_wrist_roll_joint
                1.06814150219,       // left_wrist_pitch_joint
                1.06814150219,       // left_wrist_yaw_joint
                0.907222843292423,   // right_shoulder_pitch_joint
                0.907222843292423,   // right_shoulder_roll_joint
                0.907222843292423,   // right_shoulder_yaw_joint
                0.907222843292423,   // right_elbow_joint
                0.907222843292423,   // right_wrist_roll_joint
                1.06814150219,       // right_wrist_pitch_joint
                1.06814150219        // right_wrist_yaw_joint
            };
        }

        // Default positions for 29DOF
        if (config["default_angles"])
        {
            default_angles = config["default_angles"].as<std::vector<double>>();
        }
        else
        {
            exit(1);
            // Default joint positions based on Python script
            default_angles = {
                -0.312,  // left_hip_pitch_joint
                0.0,     // left_hip_roll_joint
                0.0,     // left_hip_yaw_joint
                0.669,   // left_knee_joint
                -0.363,  // left_ankle_pitch_joint
                0.0,     // left_ankle_roll_joint
                -0.312,  // right_hip_pitch_joint
                0.0,     // right_hip_roll_joint
                0.0,     // right_hip_yaw_joint
                0.669,   // right_knee_joint
                -0.363,  // right_ankle_pitch_joint
                0.0,     // right_ankle_roll_joint
                0.0,     // waist_yaw_joint
                0.0,     // waist_roll_joint
                0.0,     // waist_pitch_joint
                0.2,     // left_shoulder_pitch_joint
                0.2,     // left_shoulder_roll_joint
                0.0,     // left_shoulder_yaw_joint
                0.6,     // left_elbow_joint
                0.0,     // left_wrist_roll_joint
                0.0,     // left_wrist_pitch_joint
                0.0,     // left_wrist_yaw_joint
                0.2,     // right_shoulder_pitch_joint
                -0.2,    // right_shoulder_roll_joint
                0.0,     // right_shoulder_yaw_joint
                0.6,     // right_elbow_joint
                0.0,     // right_wrist_roll_joint
                0.0,     // right_wrist_pitch_joint
                0.0      // right_wrist_yaw_joint
            };
        }

        // Action scaling for 29DOF
        if (config["action_scale"])
        {
            action_scale = config["action_scale"].as<std::vector<double>>();
        }
        else
        {
            exit(1);
            // Default action scales based on Python script
            action_scale = {
                0.5475464652142303,   // left_hip_pitch_joint
                0.3506614663788243,   // left_hip_roll_joint
                0.5475464652142303,   // left_hip_yaw_joint
                0.3506614663788243,   // left_knee_joint
                0.43857731392336724,  // left_ankle_pitch_joint
                0.43857731392336724,  // left_ankle_roll_joint
                0.5475464652142303,   // right_hip_pitch_joint
                0.3506614663788243,   // right_hip_roll_joint
                0.5475464652142303,   // right_hip_yaw_joint
                0.3506614663788243,   // right_knee_joint
                0.43857731392336724,  // right_ankle_pitch_joint
                0.43857731392336724,  // right_ankle_roll_joint
                0.5475464652142303,   // waist_yaw_joint
                0.43857731392336724,  // waist_roll_joint
                0.43857731392336724,  // waist_pitch_joint
                0.43857731392336724,  // left_shoulder_pitch_joint
                0.43857731392336724,  // left_shoulder_roll_joint
                0.43857731392336724,  // left_shoulder_yaw_joint
                0.43857731392336724,  // left_elbow_joint
                0.43857731392336724,  // left_wrist_roll_joint
                0.07450087032950714,  // left_wrist_pitch_joint
                0.07450087032950714,  // left_wrist_yaw_joint
                0.43857731392336724,  // right_shoulder_pitch_joint
                0.43857731392336724,  // right_shoulder_roll_joint
                0.43857731392336724,  // right_shoulder_yaw_joint
                0.43857731392336724,  // right_elbow_joint
                0.43857731392336724,  // right_wrist_roll_joint
                0.07450087032950714,  // right_wrist_pitch_joint
                0.07450087032950714   // right_wrist_yaw_joint
            };
        }

        // Model dimensions
        if (config["num_actions"])
        {
            num_actions = config["num_actions"].as<int>();
        }
        if (config["num_obs"])
        {
            num_obs = config["num_obs"].as<int>();
        }

        // Joint names for 29DOF (mujoco order)
        joint_names = {
            "left_hip_pitch_joint",      "left_hip_roll_joint",        "left_hip_yaw_joint",
            "left_knee_joint",           "left_ankle_pitch_joint",     "left_ankle_roll_joint",
            "right_hip_pitch_joint",     "right_hip_roll_joint",       "right_hip_yaw_joint",
            "right_knee_joint",          "right_ankle_pitch_joint",    "right_ankle_roll_joint",
            "waist_yaw_joint",           "waist_roll_joint",           "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",   "left_shoulder_yaw_joint",
            "left_elbow_joint",          "left_wrist_roll_joint",      "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",      "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",  "right_elbow_joint",          "right_wrist_roll_joint",
            "right_wrist_pitch_joint",   "right_wrist_yaw_joint"};

        std::cout << "Config loaded successfully for 29DOF robot" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading config file: " << e.what() << std::endl;
        throw;
    }
}
