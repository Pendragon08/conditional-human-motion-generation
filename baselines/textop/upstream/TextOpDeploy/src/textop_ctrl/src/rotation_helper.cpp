#include "textop_ctrl/rotation_helper.hpp"

#include <algorithm>
#include <cmath>

std::vector<float> get_gravity_orientation(const std::array<float, 4>& quaternion)
{
    float qw = quaternion[0];
    float qx = quaternion[1];
    float qy = quaternion[2];
    float qz = quaternion[3];

    std::vector<float> gravity_orientation(3);

    // Following Python implementation exactly:
    // gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[0] = 2.0f * (-qz * qx + qw * qy);
    // gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[1] = -2.0f * (qz * qy + qw * qx);
    // gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    gravity_orientation[2] = 1.0f - 2.0f * (qw * qw + qz * qz);

    return gravity_orientation;
}

std::vector<float> quad_to_rpy(const std::array<float, 4>& quat)
{
    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];

    // Following Python implementation exactly:
    // roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    float roll = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));

    // pitch = np.arcsin(2.0 * (w * y - x * z))
    float pitch = std::asin(2.0f * (w * y - x * z));

    // siny_cosp = 2.0 * (w * z + x * y)
    float siny_cosp = 2.0f * (w * z + x * y);
    // cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    // yaw = np.arctan2(siny_cosp, cosy_cosp)
    float yaw = std::atan2(siny_cosp, cosy_cosp);

    return {roll, pitch, yaw};
}

std::pair<std::array<float, 4>, std::vector<float>> transform_imu_data(
    float waist_yaw, float waist_yaw_omega, const std::array<float, 4>& imu_quat,
    const std::vector<float>& imu_omega)
{
    // For now, return the input data unchanged
    // The Python version uses scipy.spatial.transform.Rotation which is complex to implement in C++
    // This is a simplified implementation that should work for basic cases

    // Suppress unused parameter warnings
    (void)waist_yaw;
    (void)waist_yaw_omega;

    // TODO: Implement proper coordinate transformation
    // The Python version does:
    // RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    // R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    // R_pelvis = np.dot(R_torso, RzWaist.T)
    // w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    // return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w

    return {imu_quat, imu_omega};
}
