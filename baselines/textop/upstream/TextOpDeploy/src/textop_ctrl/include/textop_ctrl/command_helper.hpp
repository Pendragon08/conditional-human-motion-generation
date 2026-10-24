#ifndef ALMI_CTRL_COMMAND_HELPER_HPP
#define ALMI_CTRL_COMMAND_HELPER_HPP

#include <cstdint>
#include <array>
#include <vector>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_go/msg/low_cmd.hpp>

namespace almi_ctrl {

// Motor command helper functions
void init_cmd_hg(unitree_hg::msg::LowCmd& cmd, uint8_t mode_machine, uint8_t mode_pr);
void init_cmd_go(unitree_go::msg::LowCmd& cmd, bool weak_motor = false);

// Create damping command (zero torque)
void create_damping_cmd(unitree_hg::msg::LowCmd& cmd);
void create_damping_cmd(unitree_go::msg::LowCmd& cmd);

// Create zero command
void create_zero_cmd(unitree_hg::msg::LowCmd& cmd);
void create_zero_cmd(unitree_go::msg::LowCmd& cmd);

} // namespace almi_ctrl

#endif // ALMI_CTRL_COMMAND_HELPER_HPP
