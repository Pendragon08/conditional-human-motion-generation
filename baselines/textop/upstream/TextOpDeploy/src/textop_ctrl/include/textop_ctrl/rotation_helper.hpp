#pragma once

#include <array>
#include <vector>

// Gravity orientation calculation
std::vector<float> get_gravity_orientation(const std::array<float, 4>& quaternion);

// Quaternion to RPY conversion
std::vector<float> quad_to_rpy(const std::array<float, 4>& quat);

// IMU data transformation
std::pair<std::array<float, 4>, std::vector<float>> transform_imu_data(
    float waist_yaw, float waist_yaw_omega, 
    const std::array<float, 4>& imu_quat, 
    const std::vector<float>& imu_omega);
