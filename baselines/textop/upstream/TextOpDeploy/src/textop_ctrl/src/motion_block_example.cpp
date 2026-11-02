#include <rclcpp/rclcpp.hpp>

#include "textop_ctrl/motion_block_subscriber.hpp"

int main(int argc, char** argv)
{
    // Initialize ROS2
    rclcpp::init(argc, argv);

    // Create and run the subscriber node
    auto node = std::make_shared<textop_ctrl::MotionBlockSubscriber>();

    RCLCPP_INFO(node->get_logger(), "Starting MotionBlock example subscriber...");

    try
    {
        rclcpp::spin(node);
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node->get_logger(), "Exception caught: %s", e.what());
    }

    // Cleanup
    rclcpp::shutdown();

    return 0;
}