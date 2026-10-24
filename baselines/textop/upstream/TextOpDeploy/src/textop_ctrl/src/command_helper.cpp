#include "textop_ctrl/command_helper.hpp"

#include "common/motor_crc_hg.h"

namespace almi_ctrl
{

void create_damping_cmd(unitree_hg::msg::LowCmd& cmd)
{
    for (size_t i = 0; i < cmd.motor_cmd.size(); ++i)
    {
        cmd.motor_cmd[i].mode = 0;  // Disable mode
        cmd.motor_cmd[i].q = 0.0f;
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = 0.0f;
        cmd.motor_cmd[i].kd = 0.0f;
    }
}

void create_damping_cmd(unitree_go::msg::LowCmd& cmd)
{
    for (size_t i = 0; i < cmd.motor_cmd.size(); ++i)
    {
        cmd.motor_cmd[i].mode = 0;  // Disable mode
        cmd.motor_cmd[i].q = 0.0f;
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = 0.0f;
        cmd.motor_cmd[i].kd = 0.0f;
    }
}

void create_zero_cmd(unitree_hg::msg::LowCmd& cmd)
{
    for (size_t i = 0; i < cmd.motor_cmd.size(); ++i)
    {
        cmd.motor_cmd[i].mode = 1;  // Enable mode
        cmd.motor_cmd[i].q = 0.0f;
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = 0.0f;
        cmd.motor_cmd[i].kd = 0.0f;
    }
}

void create_zero_cmd(unitree_go::msg::LowCmd& cmd)
{
    for (size_t i = 0; i < cmd.motor_cmd.size(); ++i)
    {
        cmd.motor_cmd[i].mode = 1;  // Enable mode
        cmd.motor_cmd[i].q = 0.0f;
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = 0.0f;
        cmd.motor_cmd[i].kd = 0.0f;
    }
}

void init_cmd_hg(unitree_hg::msg::LowCmd& cmd, uint8_t mode_machine, uint8_t mode_pr)
{
    cmd.mode_machine = mode_machine;
    cmd.mode_pr = mode_pr;

    for (size_t i = 0; i < cmd.motor_cmd.size(); ++i)
    {
        cmd.motor_cmd[i].mode = 1;  // Enable mode
        cmd.motor_cmd[i].q = 0.0f;
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = 0.0f;
        cmd.motor_cmd[i].kd = 0.0f;
    }
}

void init_cmd_go(unitree_go::msg::LowCmd& cmd, bool /*weak_motor*/)
{
    for (size_t i = 0; i < cmd.motor_cmd.size(); ++i)
    {
        cmd.motor_cmd[i].mode = 1;  // Enable mode
        cmd.motor_cmd[i].q = 0.0f;
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = 0.0f;
        cmd.motor_cmd[i].kd = 0.0f;
    }
}

}  // namespace almi_ctrl
