#pragma once
#include <array>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <vector>

#include "motion_loader.hpp"

class ObservationComputer
{
   public:
    explicit ObservationComputer(std::shared_ptr<MotionLoader> motion_loader, rclcpp::Node* node);
    ~ObservationComputer() = default;

    // Joint reindexing arrays
    std::vector<int> mujoco_to_isaaclab_reindex_;
    std::vector<int> isaaclab_to_mujoco_reindex_;

    // Default joint angles (matching Python default_angles)
    std::vector<float> default_angles_;

    // Main observation computation
    std::vector<float> compute_observation(const unitree_hg::msg::LowState& low_state, int motion_t,
                                           const std::vector<float>& last_actions);

    // Access to reindexing arrays
    const std::vector<int>& get_isaaclab_to_mujoco_reindex() const
    {
        return isaaclab_to_mujoco_reindex_;
    }
    const std::vector<int>& get_mujoco_to_isaaclab_reindex() const
    {
        return mujoco_to_isaaclab_reindex_;
    }

    // Odometry interface (world frame)
    void set_odometry(const std::array<float, 3>& position_w,
                      const std::array<float, 3>& linear_velocity_w)
    {
        robot_pos_w_ = position_w;
        robot_linvel_w_ = linear_velocity_w;
    }

    // LockXY mode interface
    void set_lock_xy_mode(bool enabled)
    {
        if (lock_xy_mode_ != enabled)
        {
            std::cout << "LockXY mode: " << lock_xy_mode_ << " -> " << enabled << std::endl;
            if (enabled)
            {
                // Record robot XY position when entering LockXY mode
                locked_robot_xy_[0] = robot_pos_w_[0];
                locked_robot_xy_[1] = robot_pos_w_[1];
                std::cout << "Locked robot XY: [" << locked_robot_xy_[0] << ", "
                          << locked_robot_xy_[1] << "]" << std::endl;
            }
        }
        lock_xy_mode_ = enabled;
    }
    // Utility function for pretty printing
    template <typename T>
    std::string vector_to_string(const std::vector<T>& vec, int max_elements = 1000);

   private:
    // Helper functions matching Python implementation
    std::vector<float> get_command(int motion_t);
    std::vector<float> motion_anchor_pos_b_future(const unitree_hg::msg::LowState& low_state,
                                                  int motion_t);
    std::vector<float> motion_anchor_ori_b_future(const unitree_hg::msg::LowState& low_state,
                                                  int motion_t);
    std::vector<float> get_base_lin_vel(const unitree_hg::msg::LowState& low_state);
    std::vector<float> get_base_ang_vel(const unitree_hg::msg::LowState& low_state);
    std::vector<float> get_joint_pos_rel(const unitree_hg::msg::LowState& low_state);
    std::vector<float> get_joint_vel_rel(const unitree_hg::msg::LowState& low_state);
    std::vector<float> get_last_action(const std::vector<float>& last_actions);

    // Projected gravity computation
    std::vector<float> get_projected_gravity(const unitree_hg::msg::LowState& low_state);

    // Math utilities (matching Python math_np.py)
    std::array<float, 4> quat_conjugate(const std::array<float, 4>& q);
    std::array<float, 4> quat_mul(const std::array<float, 4>& q1, const std::array<float, 4>& q2);
    std::array<float, 4> quat_inv(const std::array<float, 4>& q);
    std::array<float, 3> quat_apply(const std::array<float, 4>& q, const std::array<float, 3>& v);
    std::array<float, 3> quat_rotate_inverse(const std::array<float, 4>& q,
                                             const std::array<float, 3>& v);
    std::vector<float> matrix_from_quat(const std::array<float, 4>& quat);
    std::pair<std::array<float, 3>, std::array<float, 4>> subtract_frame_transforms(
        const std::array<float, 3>& pos_a, const std::array<float, 4>& quat_a,
        const std::array<float, 3>& pos_b, const std::array<float, 4>& quat_b);

    // Frame initialization helper functions
    void setup_init_frame(const unitree_hg::msg::LowState& low_state);
    std::array<float, 4> calc_heading_quat(const std::array<float, 4>& quat);
    std::pair<std::array<float, 3>, std::array<float, 4>> transform_ref_to_robot_frame(
        const std::array<float, 3>& ref_pos, const std::array<float, 4>& ref_quat);

    std::shared_ptr<MotionLoader> motion_loader_;
    rclcpp::Node* node_;

    // Odometry state (world frame)
    std::array<float, 3> robot_pos_w_{0.0f, 0.0f, 0.8f};
    std::array<float, 3> robot_linvel_w_{0.0f, 0.0f, 0.0f};

    // Frame initialization state
    bool frame_initialized_{false};
    float robot_init_yaw_{0.0f};
    std::array<float, 3> robot_init_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 4> robot_init_quat_{1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 3> ref_init_pos_{0.0f, 0.0f, 0.0f};
    std::array<float, 4> ref_init_quat_{1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> ref_to_robot_quat_{1.0f, 0.0f, 0.0f, 0.0f};

    // LockXY mode state
    bool lock_xy_mode_{false};
    std::array<float, 2> locked_robot_xy_{0.0f,
                                          0.0f};  // Robot XY position when LockXY mode was enabled

    // Helper function to align robot and ref XY coordinates
    void align_robot_ref_xy(int motion_t);
};
