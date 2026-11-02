#include "textop_ctrl/motion_loader.hpp"

#include <iostream>

#include "textop_ctrl/msg/motion_block.hpp"

MotionLoader::MotionLoader(rclcpp::Node* node, const std::string& motion_topic) : node_(node)
{
    // Initialize body configuration
    initialize_body_configuration();

    // Initialize motion parameters
    fps = 30;  // Default FPS
    T = 0;     // Will be updated as data arrives

    // Create subscription to motion block topic
    motion_block_sub_ = node_->create_subscription<textop_ctrl::msg::MotionBlock>(
        motion_topic, 10,
        std::bind(&MotionLoader::motion_block_callback, this, std::placeholders::_1));

    RCLCPP_INFO(node_->get_logger(), "MotionLoader subscribed to topic: %s", motion_topic.c_str());
}

void MotionLoader::motion_block_callback(const textop_ctrl::msg::MotionBlock::SharedPtr msg)
{
    // Update motion sequence with new block
    update_motion_sequence(msg);

    // Update public interface variables
    T = motion_sequence_.total_length;
    fps = motion_sequence_.fps;

    // Convert internal storage to public interface format
    // joint_pos and joint_vel: direct copy
    joint_pos = motion_sequence_.joint_positions;
    joint_vel = motion_sequence_.joint_velocities;

    // body_pos and body_ori: convert from flat arrays to array format
    // For now, we only handle single body (anchor body)
    body_pos.resize(T);
    body_ori.resize(T);

    for (int t = 0; t < T; ++t)
    {
        body_pos[t].resize(1);  // Single body
        body_ori[t].resize(1);  // Single body

        if (t < static_cast<int>(motion_sequence_.anchor_body_pos.size()))
        {
            // Copy anchor body position [3]
            for (int i = 0; i < 3; ++i)
            {
                body_pos[t][0][i] = motion_sequence_.anchor_body_pos[t][i];
            }
        }

        if (t < static_cast<int>(motion_sequence_.anchor_body_ori.size()))
        {
            // Copy anchor body orientation [4]
            for (int i = 0; i < 4; ++i)
            {
                body_ori[t][0][i] = motion_sequence_.anchor_body_ori[t][i];
            }
        }
    }

    // Initialize body_ang_vel_w as empty for now (not provided in MotionBlock)
    body_ang_vel_w.resize(T);
    for (int t = 0; t < T; ++t)
    {
        body_ang_vel_w[t].resize(1);  // Single body
        for (int i = 0; i < 3; ++i)
        {
            body_ang_vel_w[t][0][i] = 0.0f;  // Default to zero
        }
    }

    // RCLCPP_DEBUG(node_->get_logger(), "Motion sequence updated: T=%d, fps=%d", T, fps);
}

void MotionLoader::update_motion_sequence(const textop_ctrl::msg::MotionBlock::SharedPtr msg)
{
    // Extract current motion data
    std::vector<float> joint_positions = extract_array_data(msg->joint_positions);
    std::vector<float> joint_velocities = extract_array_data(msg->joint_velocities);
    std::vector<float> anchor_body_ori = extract_array_data(msg->anchor_body_ori);
    std::vector<float> anchor_body_pos = extract_array_data(msg->anchor_body_pos);

    // Extract dimensions from the message
    int T_block = 0, Nq = 0;

    // Get T and Nq from joint_positions dimensions
    if (!msg->joint_positions.layout.dim.empty() && msg->joint_positions.layout.dim.size() >= 2)
    {
        T_block = msg->joint_positions.layout.dim[0].size;
        Nq = msg->joint_positions.layout.dim[1].size;
        motion_sequence_.num_joints = Nq;
    }
    else
    {
        // Fallback: infer from data size and current joint count
        if (!joint_positions.empty() && motion_sequence_.num_joints > 0)
        {
            T_block = joint_positions.size() / motion_sequence_.num_joints;
            Nq = motion_sequence_.num_joints;
        }
        else
        {
            // Last resort: try to infer T from anchor data
            if (anchor_body_ori.size() >= 4)
            {
                T_block = anchor_body_ori.size() / 4;
                Nq = motion_sequence_.num_joints;  // Use default
            }
        }
    }

    if (T_block <= 0 || Nq <= 0)
    {
        RCLCPP_WARN(node_->get_logger(),
                    "Unable to determine block dimensions (T=%d, Nq=%d), skipping update", T_block,
                    Nq);
        return;
    }

    int block_index = msg->index;
    int required_length = block_index + T_block;

    // Ensure sequence has enough capacity
    ensure_sequence_capacity(required_length);

    // Update the sequence from block_index to block_index + T_block
    for (int t = 0; t < T_block; ++t)
    {
        int seq_idx = block_index + t;

        // Update joint positions
        if (static_cast<size_t>(seq_idx) < motion_sequence_.joint_positions.size())
        {
            motion_sequence_.joint_positions[seq_idx].resize(Nq);
            for (int j = 0; j < Nq; ++j)
            {
                motion_sequence_.joint_positions[seq_idx][j] = joint_positions[t * Nq + j];
            }
        }

        // Update joint velocities
        if (static_cast<size_t>(seq_idx) < motion_sequence_.joint_velocities.size())
        {
            motion_sequence_.joint_velocities[seq_idx].resize(Nq);
            for (int j = 0; j < Nq; ++j)
            {
                motion_sequence_.joint_velocities[seq_idx][j] = joint_velocities[t * Nq + j];
            }
        }

        // Update anchor body orientation [T,4]
        if (static_cast<size_t>(seq_idx) < motion_sequence_.anchor_body_ori.size())
        {
            motion_sequence_.anchor_body_ori[seq_idx].resize(4);
            for (int i = 0; i < 4; ++i)
            {
                motion_sequence_.anchor_body_ori[seq_idx][i] = anchor_body_ori[t * 4 + i];
            }
        }

        // Update anchor body position [T,3]
        if (static_cast<size_t>(seq_idx) < motion_sequence_.anchor_body_pos.size())
        {
            motion_sequence_.anchor_body_pos[seq_idx].resize(3);
            for (int i = 0; i < 3; ++i)
            {
                motion_sequence_.anchor_body_pos[seq_idx][i] = anchor_body_pos[t * 3 + i];
            }
        }
    }

    // Update total length
    motion_sequence_.total_length = std::max(motion_sequence_.total_length, required_length);

    RCLCPP_INFO(node_->get_logger(), "Motion sequence updated: T=%d, block_size=%d",
                motion_sequence_.total_length, T_block);
}

void MotionLoader::ensure_sequence_capacity(int required_length)
{
    if (static_cast<size_t>(required_length) > motion_sequence_.joint_positions.size())
    {
        // Resize all sequences
        motion_sequence_.joint_positions.resize(required_length);
        motion_sequence_.joint_velocities.resize(required_length);
        motion_sequence_.anchor_body_ori.resize(required_length);
        motion_sequence_.anchor_body_pos.resize(required_length);

        // Initialize new elements
        for (int i = motion_sequence_.total_length; i < required_length; ++i)
        {
            motion_sequence_.joint_positions[i].resize(motion_sequence_.num_joints, 0.0f);
            motion_sequence_.joint_velocities[i].resize(motion_sequence_.num_joints, 0.0f);
            motion_sequence_.anchor_body_ori[i].resize(4, 0.0f);
            motion_sequence_.anchor_body_pos[i].resize(3, 0.0f);
        }
    }
}

std::vector<float> MotionLoader::extract_array_data(const std_msgs::msg::Float32MultiArray& array)
{
    std::vector<float> data;
    data.reserve(array.data.size());

    for (const auto& value : array.data)
    {
        data.push_back(value);
    }

    return data;
}

void MotionLoader::initialize_body_configuration()
{
    // Initialize body names and anchor body (same as original)
    body_names = {
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    };
    anchor_body_name = "pelvis";
    anchor_body_index = 0;
}
