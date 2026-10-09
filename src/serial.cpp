#include "serial.hpp"
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

double monotonic_time_ms()
{
    static const auto origin = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - origin).count();
}

static std::uint8_t serial_crc8(const std::uint8_t *bytes, std::size_t size)
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

Serial::~Serial()
{
    try { close(); } catch (...) {}
}

void Serial::open(const std::string &device)
{
    close();
#ifdef _WIN32
    const auto path = device.rfind("\\\\.\\", 0) == 0 ? device : "\\\\.\\" + device;
    HANDLE opened = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (opened == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot open " + device + " (Win32 " + std::to_string(GetLastError()) + ")");
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
    try { worker = std::thread(&Serial::run, this); }
    catch (...) { close(); throw; }
#else
    throw std::runtime_error("Serial hardware transport requires Windows");
#endif
}

void Serial::close()
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

bool Serial::receive(GimbalState &state)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!error.empty())
        throw std::runtime_error(error);
    if (!has_state)
        return false;
    state = latest;
    has_state = false;
    return true;
}

void Serial::send_target(float yaw_rad, float pitch_rad, bool valid)
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
    packet.push_back(serial_crc8(packet.data() + 1, packet.size() - 1));
    std::lock_guard<std::mutex> lock(mutex);
    if (!running)
        throw std::runtime_error(error.empty() ? "Serial port is not open" : error);
    pending_target = std::move(packet);
}

void Serial::send_mode(std::uint8_t mode)
{
    std::vector<std::uint8_t> packet{0xA5, HOST_MODE, 1, mode};
    packet.push_back(serial_crc8(packet.data() + 1, packet.size() - 1));
    std::lock_guard<std::mutex> lock(mutex);
    if (!running)
        throw std::runtime_error(error.empty() ? "Serial port is not open" : error);
    pending_mode = std::move(packet);
}

void Serial::run()
{
#ifdef _WIN32
    try
    {
        std::vector<std::uint8_t> buffer;
        std::uint8_t bytes[256];
        double next_heartbeat_ms = 0;
        std::vector<std::uint8_t> heartbeat{0xA5, HOST_HEARTBEAT, 0};
        heartbeat.push_back(serial_crc8(heartbeat.data() + 1, 2));
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
                if (monotonic_time_ms() >= next_heartbeat_ms)
                {
                    write_packet(heartbeat);
                    next_heartbeat_ms = monotonic_time_ms() + 50;
                }
                DWORD count = 0, errors = 0;
                if (!ClearCommError(handle, &errors, nullptr) || !ReadFile(handle, bytes, sizeof(bytes), &count, nullptr))
                    throw std::runtime_error("Serial read failed (Win32 " + std::to_string(GetLastError()) + ")");
                buffer.insert(buffer.end(), bytes, bytes + count);
                GimbalState state;
                bool received = false;
                while (buffer.size() >= 18)
                {
                    if (buffer[0] != 0xA5 || buffer[1] != HOST_STATE || buffer[2] != 14)
                        throw std::runtime_error("Invalid MCU STATE packet header");
                    if (serial_crc8(buffer.data() + 1, 16) != buffer[17])
                        throw std::runtime_error("MCU STATE CRC mismatch");
                    float *angles[] = {&state.yaw_rad, &state.pitch_rad, &state.roll_rad};
                    for (unsigned angle = 0; angle < 3; ++angle)
                    {
                        std::uint32_t bits = 0;
                        for (unsigned byte = 0; byte < 4; ++byte)
                            bits |= static_cast<std::uint32_t>(buffer[3 + angle * 4 + byte]) << (byte * 8);
                        std::memcpy(angles[angle], &bits, sizeof(bits));
                    }
                    state.mode = buffer[15];
                    state.flags = buffer[16];
                    state.receive_timestamp_ms = monotonic_time_ms();
                    received = true;
                    buffer.erase(buffer.begin(), buffer.begin() + 18);
                }
                if (received)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    latest = state;
                    has_state = true;
                }
            }
            for (bool target : {false, true})
            {
                std::vector<std::uint8_t> packet;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    packet.swap(target ? pending_target : pending_mode);
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
