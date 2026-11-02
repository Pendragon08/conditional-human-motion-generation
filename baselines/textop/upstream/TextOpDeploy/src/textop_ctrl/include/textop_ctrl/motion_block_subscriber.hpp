#pragma once

#include <cnpy.h>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <string>
#include <vector>

#include "textop_ctrl/msg/motion_block.hpp"

namespace textop_ctrl
{

class MotionBlockSubscriber : public rclcpp::Node
{
   public:
    MotionBlockSubscriber();
    ~MotionBlockSubscriber() = default;

   private:
    // Callback function for MotionBlock messages
    void motion_block_callback(const textop_ctrl::msg::MotionBlock::SharedPtr msg);

    // Helper functions to extract data from Float32MultiArray
    std::vector<float> extract_array_data(const std_msgs::msg::Float32MultiArray& array);
    void print_array_info(const std_msgs::msg::Float32MultiArray& array, const std::string& name);

    // Convert ROS time to timestamp
    double time_to_seconds(const builtin_interfaces::msg::Time& time_msg);

    // Motion sequence management
    void update_motion_sequence(const textop_ctrl::msg::MotionBlock::SharedPtr msg);
    void save_motion_to_npz(const std::string& filename);
    void ensure_sequence_capacity(int required_length);

    // Subscriber
    rclcpp::Subscription<textop_ctrl::msg::MotionBlock>::SharedPtr motion_block_sub_;

    // Data storage for current block
    struct MotionData
    {
        int32_t index;
        double timestamp;
        std::vector<float> joint_positions;   // [T, Nq]
        std::vector<float> joint_velocities;  // [T, Nq]
        std::vector<float> anchor_body_ori;   // [T, 4]
        std::vector<float> anchor_body_pos;   // [T, 3]
    };

    // Complete motion sequence storage
    struct MotionSequence
    {
        std::vector<std::vector<float>> joint_positions;   // [Total_T, Nq]
        std::vector<std::vector<float>> joint_velocities;  // [Total_T, Nq]
        std::vector<std::vector<float>> anchor_body_ori;   // [Total_T, 4]
        std::vector<std::vector<float>> anchor_body_pos;   // [Total_T, 3]
        int total_length = 0;
        int num_joints = 29;  // Default Nq
        int fps = 50;         // Default FPS
    };

    MotionData current_motion_data_;
    MotionSequence motion_sequence_;
    int block_counter_ = 0;  // For generating unique filenames
};

}  // namespace textop_ctrl