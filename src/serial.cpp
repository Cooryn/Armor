#include "serial.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "Protocol requires IEEE-754 float32");

double serial::monotonic_time_ms()
{
    static const auto origin = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - origin).count();
}

std::uint8_t serial::crc8(const std::uint8_t *bytes, std::size_t size)
{
    std::uint8_t crc = 0;
    for (std::size_t i = 0; i < size; ++i)
    {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = static_cast<std::uint8_t>((crc & 0x80) ? (crc << 1) ^ 0x07 : crc << 1);
    }
    return crc;
}

serial::~serial()
{
    try { close(); } catch (...) {}
}

void serial::open(const std::string &serial_port)
{
    close();
#ifdef _WIN32
    const auto path = serial_port.rfind("\\\\.\\", 0) == 0 ? serial_port : "\\\\.\\" + serial_port;
    HANDLE opened = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (opened == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot open " + serial_port + " (Win32 " + std::to_string(GetLastError()) + ")");
    DCB settings{};
    settings.DCBlength = sizeof(settings);
    settings.BaudRate = 921600;
    settings.ByteSize = 8;
    settings.StopBits = ONESTOPBIT;
    settings.Parity = NOPARITY;
    settings.fBinary = TRUE;
    settings.XonChar = 0x11;
    settings.XoffChar = 0x13;
    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = MAXDWORD;
    timeouts.ReadTotalTimeoutMultiplier = MAXDWORD;
    timeouts.ReadTotalTimeoutConstant = 5;
    timeouts.WriteTotalTimeoutConstant = 20;
    if (!SetCommState(opened, &settings) || !SetCommTimeouts(opened, &timeouts) ||
        !PurgeComm(opened, PURGE_RXCLEAR | PURGE_TXCLEAR))
    {
        const auto code = GetLastError();
        CloseHandle(opened);
        throw std::runtime_error("Serial configuration failed (Win32 " + std::to_string(code) + ")");
    }
    handle = opened;
    running = true;
    try { worker = std::thread(&serial::run, this); }
    catch (...) { close(); throw; }
#else
    throw std::runtime_error("Serial hardware transport requires Windows");
#endif
}

void serial::close()
{
    running = false;
    if (worker.joinable())
        worker.join();
#ifdef _WIN32
    if (handle)
        CloseHandle(handle);
#endif
    std::lock_guard<std::mutex> lock(mutex);
    handle = nullptr;
    has_state = false;
    pending_target.clear();
    pending_mode.clear();
    std::string failure;
    failure.swap(error);
    if (!failure.empty())
        throw std::runtime_error(failure);
}

bool serial::receive(GimbalState &gimbal_state)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!error.empty())
        throw std::runtime_error(error);
    if (!has_state)
        return false;
    gimbal_state = latest_state;
    has_state = false;
    return true;
}

void serial::send_target(float yaw_rad, float pitch_rad, bool valid)
{
    std::vector<std::uint8_t> packet{0xA5, HOST_TARGET, 9};
    for (float angle : {yaw_rad, pitch_rad})
    {
        std::uint32_t bits;
        std::memcpy(&bits, &angle, sizeof(bits));
        for (unsigned byte = 0; byte < 4; ++byte)
            packet.push_back(static_cast<std::uint8_t>(bits >> (byte * 8)));
    }
    packet.push_back(valid ? 1 : 0);
    packet.push_back(serial::crc8(packet.data() + 1, packet.size() - 1));
    std::lock_guard<std::mutex> lock(mutex);
    if (!running)
        throw std::runtime_error(error.empty() ? "Serial port is not open" : error);
    pending_target = std::move(packet);
    target_updated_ms = serial::monotonic_time_ms();
}

void serial::send_mode(std::uint8_t mode)
{
    std::vector<std::uint8_t> packet{0xA5, HOST_MODE, 1, mode};
    packet.push_back(serial::crc8(packet.data() + 1, packet.size() - 1));
    std::lock_guard<std::mutex> lock(mutex);
    if (!running)
        throw std::runtime_error(error.empty() ? "Serial port is not open" : error);
    pending_mode = std::move(packet);
}

void serial::run()
{
#ifdef _WIN32
    try
    {
        std::vector<std::uint8_t> buffer;
        std::uint8_t bytes[256];
        double next_heartbeat_ms = 0;
        double next_target_ms = 0;
        std::vector<std::uint8_t> target_packet{0xA5, HOST_TARGET, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x3E};
        double target_time_ms = 0;
        std::vector<std::uint8_t> heartbeat{0xA5, HOST_HEARTBEAT, 0};
        heartbeat.push_back(serial::crc8(heartbeat.data() + 1, 2));
        const auto write_packet = [this](const std::vector<std::uint8_t> &packet)
        {
            DWORD sent = 0;
            if (!WriteFile(handle, packet.data(), static_cast<DWORD>(packet.size()), &sent, nullptr) || sent != packet.size())
                throw std::runtime_error("Serial write failed or timed out (Win32 " + std::to_string(GetLastError()) + ")");
        };
        while (true)
        {
            if (running)
            {
                if (serial::monotonic_time_ms() >= next_heartbeat_ms)
                {
                    write_packet(heartbeat);
                    next_heartbeat_ms = serial::monotonic_time_ms() + 50;
                }
                DWORD count = 0, errors = 0;
                if (!ClearCommError(handle, &errors, nullptr) || !ReadFile(handle, bytes, sizeof(bytes), &count, nullptr))
                    throw std::runtime_error("Serial read failed (Win32 " + std::to_string(GetLastError()) + ")");
                buffer.insert(buffer.end(), bytes, bytes + count);
                GimbalState gimbal_state;
                bool received = false;
                while (!buffer.empty())
                {
                    buffer.erase(buffer.begin(), std::find(buffer.begin(), buffer.end(), 0xA5));
                    if (buffer.size() < 3)
                        break;
                    const auto cmd = buffer[1];
                    const auto length = buffer[2];
                    if (length > 24 || (cmd == HOST_STATE && length != 14) ||
                        (cmd == HOST_HEARTBEAT && length != 0))
                    {
                        buffer.erase(buffer.begin());
                        continue;
                    }
                    const std::size_t frame_size = length + 4;
                    if (buffer.size() < frame_size)
                        break;
                    if (serial::crc8(buffer.data() + 1, length + 2) != buffer[frame_size - 1])
                    {
                        buffer.erase(buffer.begin());
                        continue;
                    }
                    // HEARTBEAT and other commands do not publish or refresh a STATE sample.
                    if (cmd != HOST_STATE)
                    {
                        buffer.erase(buffer.begin(), buffer.begin() + frame_size);
                        continue;
                    }
                    float *angles[] = {&gimbal_state.yaw_rad, &gimbal_state.pitch_rad, &gimbal_state.roll_rad};
                    for (unsigned angle = 0; angle < 3; ++angle)
                    {
                        std::uint32_t bits = 0;
                        for (unsigned byte = 0; byte < 4; ++byte)
                            bits |= static_cast<std::uint32_t>(buffer[3 + angle * 4 + byte]) << (byte * 8);
                        std::memcpy(angles[angle], &bits, sizeof(bits));
                    }
                    gimbal_state.mode = buffer[15];
                    gimbal_state.flags = buffer[16];
                    // Protocol v1.0 reserves roll; do not use it as measured attitude.
                    gimbal_state.roll_rad = 0;
                    gimbal_state.receive_timestamp_ms = serial::monotonic_time_ms();
                    received = true;
                    buffer.erase(buffer.begin(), buffer.begin() + 18);
                }
                if (received)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    latest_state = gimbal_state;
                    has_state = true;
                }
            }
            for (bool target : {false, true})
            {
                std::vector<std::uint8_t> packet;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (target && running && serial::monotonic_time_ms() < next_target_ms)
                        continue;
                    packet.swap(target ? pending_target : pending_mode);
                    if (target)
                    {
                        if (!packet.empty())
                        {
                            target_packet = packet;
                            target_time_ms = target_updated_ms;
                        }
                        if (running)
                            packet = target_packet;
                        // Repetition must never make an old vision result valid again.
                        if (!packet.empty() && serial::monotonic_time_ms() - target_time_ms >= 100)
                        {
                            packet[11] = 0;
                            packet[12] = serial::crc8(packet.data() + 1, 11);
                        }
                        next_target_ms = serial::monotonic_time_ms() + 25; // 40 Hz
                    }
                }
                if (!packet.empty())
                    write_packet(packet);
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (!running && pending_mode.empty() && pending_target.empty())
                break;
        }
    }
    catch (const std::exception &failure)
    {
        std::lock_guard<std::mutex> lock(mutex);
        error = failure.what();
        running = false;
        has_state = false;
        pending_target.clear();
        pending_mode.clear();
    }
#endif
}
