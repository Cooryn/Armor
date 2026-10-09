#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

double monotonic_time_ms(); // Common PC clock for live serial/camera timestamps.

enum HostCommand : std::uint8_t { HOST_TARGET = 1, HOST_STATE = 2, HOST_MODE = 3, HOST_HEARTBEAT = 4 };
enum HostMode : std::uint8_t { HOST_IDLE = 0, HOST_MANUAL = 1, HOST_AUTO_AIM = 2, HOST_STABILIZE = 3 };

struct GimbalState
{
    double receive_timestamp_ms = 0;
    float yaw_rad = 0, pitch_rad = 0, roll_rad = 0;
    std::uint8_t mode = 0, flags = 0; // STATE.flags is reserved in the current firmware.
};

class Serial
{
public:
    ~Serial();
    void open(const std::string &device); // Windows, 921600 baud, 8N1; owner-thread operation.
    void close(); // Drain queued writes, join the worker and close; reports communication errors.
    bool receive(GimbalState &state); // Returns only the latest new STATE.
    void send_target(float yaw_rad, float pitch_rad, bool valid);
    void send_mode(std::uint8_t mode);
private:
    void run();
    void *handle = nullptr; // Windows HANDLE; avoids Windows headers in the interface.
    std::thread worker;
    std::atomic<bool> running{false};
    std::mutex mutex;
    GimbalState latest;
    bool has_state = false;
    std::vector<std::uint8_t> pending_target, pending_mode;
    std::string error;
};
