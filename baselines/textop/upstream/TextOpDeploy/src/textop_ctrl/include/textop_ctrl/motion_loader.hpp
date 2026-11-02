#pragma once

#include <cnpy.h>

#include <array>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <string>
#include <vector>

#include "textop_ctrl/msg/motion_block.hpp"

class MotionLoader
{
   public:
    // Modified constructor to accept ROS node and topic name
    explicit MotionLoader(rclcpp::Node* node, const std::string& motion_topic = "/dar/motion");
    ~MotionLoader() = default;

    // Motion data - keeping the same interface
    std::vector<std::vector<float>> joint_pos;                      // [T, 29]
    std::vector<std::vector<float>> joint_vel;                      // [T, 29]
    std::vector<std::vector<std::array<float, 3>>> body_pos;        // [T, N, 3]
    std::vector<std::vector<std::array<float, 4>>> body_ori;        // [T, N, 4]
    std::vector<std::vector<std::array<float, 3>>> body_ang_vel_w;  // [T, N, 3]

    // Motion parameters
    int fps;
    int T;  // Total time steps

    // Body configuration
    std::vector<std::string> body_names;
    std::string anchor_body_name;
    int anchor_body_index;
    int future_steps = 5;

    // New method to check if motion data is available
    bool has_motion_data() const { return T > 0; }

   private:
    // Topic-based data loading
    void motion_block_callback(const textop_ctrl::msg::MotionBlock::SharedPtr msg);
    void update_motion_sequence(const textop_ctrl::msg::MotionBlock::SharedPtr msg);
    void ensure_sequence_capacity(int required_length);
    std::vector<float> extract_array_data(const std_msgs::msg::Float32MultiArray& array);
    void initialize_body_configuration();

    // ROS2 components
    rclcpp::Node* node_;
    rclcpp::Subscription<textop_ctrl::msg::MotionBlock>::SharedPtr motion_block_sub_;

    // Internal data storage
    struct MotionSequence
    {
        std::vector<std::vector<float>> joint_positions;   // [Total_T, Nq]
        std::vector<std::vector<float>> joint_velocities;  // [Total_T, Nq]
        std::vector<std::vector<float>> anchor_body_ori;   // [Total_T, 4]
        std::vector<std::vector<float>> anchor_body_pos;   // [Total_T, 3]
        int total_length = 0;
        int num_joints = 29;  // Default Nq
        int fps = 30;         // Default FPS
    };

    MotionSequence motion_sequence_;
};
