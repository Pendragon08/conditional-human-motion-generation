#include "textop_ctrl/remote_controller.hpp"

#include <cstring>

RemoteController::RemoteController()
{
    button.fill(0);
    lx = ly = rx = ry = 0.0f;
}

void RemoteController::set(const std::array<uint8_t, 40>& wireless_remote)
{
    update_gamepad(wireless_remote);
}

void RemoteController::update_gamepad(const std::array<uint8_t, 40>& data)
{
    // Parse gamepad data from wireless remote
    // This is a simplified implementation based on the Python version

    // Extract button states (simplified)
    // In practice, you would parse the actual gamepad protocol
    for (int i = 0; i < 16; i++)
    {
        button[i] = (data[i / 8] >> (i % 8)) & 1;
    }

    // Extract analog stick values
    // These are simplified extractions - actual implementation would depend on protocol
    if (data.size() >= 24)
    {
        // Left stick X (lx)
        int16_t lx_raw = static_cast<int16_t>((data[20] << 8) | data[21]);
        lx = static_cast<float>(lx_raw) / 32768.0f;

        // Left stick Y (ly)
        int16_t ly_raw = static_cast<int16_t>((data[22] << 8) | data[23]);
        ly = static_cast<float>(ly_raw) / 32768.0f;

        // Right stick X (rx)
        int16_t rx_raw = static_cast<int16_t>((data[24] << 8) | data[25]);
        rx = static_cast<float>(rx_raw) / 32768.0f;

        // Right stick Y (ry)
        int16_t ry_raw = static_cast<int16_t>((data[26] << 8) | data[27]);
        ry = static_cast<float>(ry_raw) / 32768.0f;
    }
}
