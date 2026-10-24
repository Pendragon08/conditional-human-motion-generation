#include "textop_ctrl/motion_block_subscriber.hpp"

#include <sys/stat.h>

#include <filesystem>
#include <iomanip>
#include <iostream>

namespace textop_ctrl
{

MotionBlockSubscriber::MotionBlockSubscriber() : Node("motion_block_subscriber")
{
    // Create subscription
    motion_block_sub_ = this->create_subscription<textop_ctrl::msg::MotionBlock>(
        "/dar/motion", 10,
        std::bind(&MotionBlockSubscriber::motion_block_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "MotionBlock Subscriber started, waiting for messages...");
}

void MotionBlockSubscriber::motion_block_callback(
    const textop_ctrl::msg::MotionBlock::SharedPtr msg)
{
    // Store basic information
    current_motion_data_.index = msg->index;
    current_motion_data_.timestamp = time_to_seconds(msg->timestamp);

    // Extract array data
    current_motion_data_.joint_positions = extract_array_data(msg->joint_positions);
    current_motion_data_.joint_velocities = extract_array_data(msg->joint_velocities);
    current_motion_data_.anchor_body_ori = extract_array_data(msg->anchor_body_ori);
    current_motion_data_.anchor_body_pos = extract_array_data(msg->anchor_body_pos);

    // Update motion sequence with new block
    update_motion_sequence(msg);

    // Print received data summary (keeping original print logic)
    RCLCPP_INFO(this->get_logger(), "=== Received MotionBlock #%d ===", current_motion_data_.index);
    RCLCPP_INFO(this->get_logger(), "Timestamp: %.6f seconds", current_motion_data_.timestamp);

    // Print array information
    print_array_info(msg->joint_positions, "Joint Positions [T,Nq]");
    print_array_info(msg->joint_velocities, "Joint Velocities [T,Nq]");
    print_array_info(msg->anchor_body_ori, "Anchor Body Orientation [T,4]");
    print_array_info(msg->anchor_body_pos, "Anchor Body Position [T,3]");

    // Example: Print some specific values from the arrays
    if (!current_motion_data_.joint_positions.empty())
    {
        // Assuming T=10, Nq=29, so first time step joint positions are indices 0-28
        RCLCPP_INFO(
            this->get_logger(), "First time step, first 3 joint positions: [%.3f, %.3f, %.3f]",
            current_motion_data_.joint_positions[0], current_motion_data_.joint_positions[1],
            current_motion_data_.joint_positions[2]);
    }

    if (current_motion_data_.anchor_body_ori.size() >= 4)
    {
        // First quaternion (first time step)
        RCLCPP_INFO(
            this->get_logger(),
            "First time step anchor orientation (w,x,y,z): [%.3f, %.3f, %.3f, %.3f]",
            current_motion_data_.anchor_body_ori[0], current_motion_data_.anchor_body_ori[1],
            current_motion_data_.anchor_body_ori[2], current_motion_data_.anchor_body_ori[3]);
    }

    if (current_motion_data_.anchor_body_pos.size() >= 3)
    {
        // First position (first time step)
        RCLCPP_INFO(
            this->get_logger(), "First time step anchor position (x,y,z): [%.3f, %.3f, %.3f]",
            current_motion_data_.anchor_body_pos[0], current_motion_data_.anchor_body_pos[1],
            current_motion_data_.anchor_body_pos[2]);
    }

    // Save motion sequence to NPZ file
    std::string filename = "log_motion/motion_sequence.npz";
    // Make directory if not exist
    {
        std::string dir = "log_motion";
        struct stat st;
        if (stat(dir.c_str(), &st) != 0)
        {
            mkdir(dir.c_str(), 0755);
        }
    }
    save_motion_to_npz(filename);
    block_counter_++;

    RCLCPP_INFO(this->get_logger(), "Motion sequence length: %d, saved to: %s",
                motion_sequence_.total_length, filename.c_str());

    std::cout << std::endl;  // Add spacing between messages
}

void MotionBlockSubscriber::update_motion_sequence(
    const textop_ctrl::msg::MotionBlock::SharedPtr msg)
{
    // Extract dimensions from the message
    int T = 0, Nq = 0;

    // Get T and Nq from joint_positions dimensions
    if (!msg->joint_positions.layout.dim.empty() && msg->joint_positions.layout.dim.size() >= 2)
    {
        T = msg->joint_positions.layout.dim[0].size;
        Nq = msg->joint_positions.layout.dim[1].size;
        motion_sequence_.num_joints = Nq;
    }
    else
    {
        // Fallback: infer from data size and current joint count
        if (!current_motion_data_.joint_positions.empty() && motion_sequence_.num_joints > 0)
        {
            T = current_motion_data_.joint_positions.size() / motion_sequence_.num_joints;
            Nq = motion_sequence_.num_joints;
        }
        else
        {
            // Last resort: try to infer T from anchor data
            if (current_motion_data_.anchor_body_ori.size() >= 4)
            {
                T = current_motion_data_.anchor_body_ori.size() / 4;
                Nq = motion_sequence_.num_joints;  // Use default
            }
        }
    }

    if (T <= 0 || Nq <= 0)
    {
        RCLCPP_WARN(this->get_logger(),
                    "Unable to determine block dimensions (T=%d, Nq=%d), skipping update", T, Nq);
        return;
    }

    int block_index = msg->index;
    int required_length = block_index + T;

    // Ensure sequence has enough capacity
    ensure_sequence_capacity(required_length);

    // Update the sequence from block_index to block_index + T
    for (int t = 0; t < T; ++t)
    {
        int seq_idx = block_index + t;

        // Update joint positions
        if (static_cast<size_t>(seq_idx) < motion_sequence_.joint_positions.size())
        {
            motion_sequence_.joint_positions[seq_idx].resize(Nq);
            for (int j = 0; j < Nq; ++j)
            {
                motion_sequence_.joint_positions[seq_idx][j] =
                    current_motion_data_.joint_positions[t * Nq + j];
            }
        }

        // Update joint velocities
        if (static_cast<size_t>(seq_idx) < motion_sequence_.joint_velocities.size())
        {
            motion_sequence_.joint_velocities[seq_idx].resize(Nq);
            for (int j = 0; j < Nq; ++j)
            {
                motion_sequence_.joint_velocities[seq_idx][j] =
                    current_motion_data_.joint_velocities[t * Nq + j];
            }
        }

        // Update anchor body orientation [T,4]
        if (static_cast<size_t>(seq_idx) < motion_sequence_.anchor_body_ori.size())
        {
            motion_sequence_.anchor_body_ori[seq_idx].resize(4);
            for (int i = 0; i < 4; ++i)
            {
                motion_sequence_.anchor_body_ori[seq_idx][i] =
                    current_motion_data_.anchor_body_ori[t * 4 + i];
            }
        }

        // Update anchor body position [T,3]
        if (static_cast<size_t>(seq_idx) < motion_sequence_.anchor_body_pos.size())
        {
            motion_sequence_.anchor_body_pos[seq_idx].resize(3);
            for (int i = 0; i < 3; ++i)
            {
                motion_sequence_.anchor_body_pos[seq_idx][i] =
                    current_motion_data_.anchor_body_pos[t * 3 + i];
            }
        }
    }

    // Update total length
    motion_sequence_.total_length = std::max(motion_sequence_.total_length, required_length);
}

void MotionBlockSubscriber::ensure_sequence_capacity(int required_length)
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

void MotionBlockSubscriber::save_motion_to_npz(const std::string& filename)
{
    if (motion_sequence_.total_length == 0)
    {
        RCLCPP_WARN(this->get_logger(), "No motion data to save");
        return;
    }

    try
    {
        // Prepare data for saving - flatten the 2D vectors
        std::vector<float> joint_pos_flat;
        std::vector<float> joint_vel_flat;
        std::vector<float> anchor_ori_flat;
        std::vector<float> anchor_pos_flat;

        joint_pos_flat.reserve(motion_sequence_.total_length * motion_sequence_.num_joints);
        joint_vel_flat.reserve(motion_sequence_.total_length * motion_sequence_.num_joints);
        anchor_ori_flat.reserve(motion_sequence_.total_length * 4);
        anchor_pos_flat.reserve(motion_sequence_.total_length * 3);

        for (int t = 0; t < motion_sequence_.total_length; ++t)
        {
            // Joint positions and velocities
            for (int j = 0; j < motion_sequence_.num_joints; ++j)
            {
                joint_pos_flat.push_back(motion_sequence_.joint_positions[t][j]);
                joint_vel_flat.push_back(motion_sequence_.joint_velocities[t][j]);
            }

            // Anchor body orientation (4 elements)
            for (int i = 0; i < 4; ++i)
            {
                anchor_ori_flat.push_back(motion_sequence_.anchor_body_ori[t][i]);
            }

            // Anchor body position (3 elements)
            for (int i = 0; i < 3; ++i)
            {
                anchor_pos_flat.push_back(motion_sequence_.anchor_body_pos[t][i]);
            }
        }

        // Save to NPZ file with same structure as motion_loader.cpp
        std::vector<size_t> joint_shape = {static_cast<size_t>(motion_sequence_.total_length),
                                           static_cast<size_t>(motion_sequence_.num_joints)};
        std::vector<size_t> anchor_ori_shape = {static_cast<size_t>(motion_sequence_.total_length),
                                                1, 4};  // [T, 1, 4] for single anchor body
        std::vector<size_t> anchor_pos_shape = {static_cast<size_t>(motion_sequence_.total_length),
                                                1, 3};  // [T, 1, 3] for single anchor body
        std::vector<size_t> fps_shape = {1};

        cnpy::npz_save(filename, "joint_pos", joint_pos_flat.data(), joint_shape, "w");
        cnpy::npz_save(filename, "joint_vel", joint_vel_flat.data(), joint_shape, "a");
        cnpy::npz_save(filename, "body_quat_w", anchor_ori_flat.data(), anchor_ori_shape, "a");
        cnpy::npz_save(filename, "body_pos_w", anchor_pos_flat.data(), anchor_pos_shape, "a");
        cnpy::npz_save(filename, "fps", &motion_sequence_.fps, fps_shape, "a");
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error saving NPZ file: %s", e.what());
    }
}

std::vector<float> MotionBlockSubscriber::extract_array_data(
    const std_msgs::msg::Float32MultiArray& array)
{
    std::vector<float> data;
    data.reserve(array.data.size());

    for (const auto& value : array.data)
    {
        data.push_back(value);
    }

    return data;
}

void MotionBlockSubscriber::print_array_info(const std_msgs::msg::Float32MultiArray& array,
                                             const std::string& name)
{
    std::cout << name << " - Size: " << array.data.size();

    // Print dimension information
    if (!array.layout.dim.empty())
    {
        std::cout << ", Dimensions: ";
        for (size_t i = 0; i < array.layout.dim.size(); ++i)
        {
            if (i > 0)
                std::cout << " x ";
            std::cout << array.layout.dim[i].label << "(" << array.layout.dim[i].size << ")";
        }
    }

    // Print first few values as example
    std::cout << ", First 3 values: [";
    for (size_t i = 0; i < std::min(static_cast<size_t>(3), array.data.size()); ++i)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) << array.data[i];
    }
    if (array.data.size() > 3)
    {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

double MotionBlockSubscriber::time_to_seconds(const builtin_interfaces::msg::Time& time_msg)
{
    return static_cast<double>(time_msg.sec) + static_cast<double>(time_msg.nanosec) * 1e-9;
}

}  // namespace textop_ctrl