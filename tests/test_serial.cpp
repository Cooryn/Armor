#include "serial.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

static std::mutex transport_mutex;
static std::deque<std::vector<std::uint8_t>> incoming;
static std::vector<std::vector<std::uint8_t>> outgoing;
static std::atomic<int> consumed{0}, closed{0};
static std::atomic<bool> fail_write{false}, fail_read{false};

static HANDLE test_open(LPCSTR name, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE)
{
    if (std::string(name) != "\\\\.\\TEST")
    {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return INVALID_HANDLE_VALUE;
    }
    return reinterpret_cast<HANDLE>(1);
}
static BOOL test_settings(HANDLE, LPDCB dcb)
{
    check(dcb->BaudRate == 921600 && dcb->ByteSize == 8 && dcb->StopBits == ONESTOPBIT &&
          dcb->Parity == NOPARITY && dcb->XonChar != dcb->XoffChar, "921600 8N1 configuration");
    return TRUE;
}
static BOOL test_timeouts(HANDLE, LPCOMMTIMEOUTS timeouts)
{
    check(timeouts->ReadTotalTimeoutConstant == 5 && timeouts->WriteTotalTimeoutConstant == 20,
          "bounded I/O timeouts");
    return TRUE;
}
static BOOL test_purge(HANDLE, DWORD) { return TRUE; }
static BOOL test_clear(HANDLE, LPDWORD errors, LPCOMSTAT) { *errors = 0; return TRUE; }
static BOOL test_close(HANDLE) { ++closed; return TRUE; }
static BOOL test_read(HANDLE, LPVOID bytes, DWORD capacity, LPDWORD count, LPOVERLAPPED)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (fail_read.exchange(false)) { SetLastError(ERROR_READ_FAULT); return FALSE; }
    std::lock_guard<std::mutex> lock(transport_mutex);
    *count = 0;
    if (!incoming.empty())
    {
        const auto &chunk = incoming.front();
        check(chunk.size() <= capacity, "test chunk fits transport buffer");
        *count = static_cast<DWORD>(chunk.size());
        if (*count) std::memcpy(bytes, chunk.data(), *count);
        incoming.pop_front();
        ++consumed;
    }
    return TRUE;
}
static BOOL test_write(HANDLE, LPCVOID bytes, DWORD count, LPDWORD sent, LPOVERLAPPED)
{
    if (fail_write.exchange(false)) { SetLastError(ERROR_WRITE_FAULT); return FALSE; }
    std::lock_guard<std::mutex> lock(transport_mutex);
    const auto *data = static_cast<const std::uint8_t *>(bytes);
    outgoing.emplace_back(data, data + count);
    *sent = count;
    return TRUE;
}
#define CreateFileA test_open
#define SetCommState test_settings
#define SetCommTimeouts test_timeouts
#define PurgeComm test_purge
#define ClearCommError test_clear
#define CloseHandle test_close
#define ReadFile test_read
#define WriteFile test_write
#endif
#include "../src/serial.cpp"
#ifdef _WIN32
#undef CreateFileA
#undef SetCommState
#undef SetCommTimeouts
#undef PurgeComm
#undef ClearCommError
#undef CloseHandle
#undef ReadFile
#undef WriteFile

static void reset_transport()
{
    std::lock_guard<std::mutex> lock(transport_mutex);
    incoming.clear(); outgoing.clear(); consumed = 0; closed = 0;
    fail_write = false; fail_read = false;
}
static void feed(std::vector<std::uint8_t> bytes)
{
    std::lock_guard<std::mutex> lock(transport_mutex);
    incoming.push_back(std::move(bytes));
}
static GimbalState await_state(::serial &link)
{
    GimbalState state;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!link.receive(state))
    {
        check(std::chrono::steady_clock::now() < deadline, "STATE reception timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return state;
}
#endif

int main()
{
    try
    {
        const std::string text = "123456789";
        check(serial::crc8(reinterpret_cast<const std::uint8_t *>(text.data()), text.size()) == 0xF4,
              "CRC-8/ATM reference vector");
#ifdef _WIN32
        const std::vector<std::uint8_t> packet{
            0xA5, 2, 14, 0, 0, 0x80, 0x3F, 0, 0, 0, 0xBF, 0, 0, 0, 0, 2, 0, 0xDA};
        for (std::size_t split = 0; split < packet.size(); ++split)
        {
            reset_transport();
            ::serial link;
            link.open("TEST");
            feed({packet.begin(), packet.begin() + split});
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (!consumed)
            {
                check(std::chrono::steady_clock::now() < deadline, "partial read timeout");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            GimbalState state;
            check(!link.receive(state), "partial STATE emits nothing");
            const auto before = serial::monotonic_time_ms();
            feed({packet.begin() + split, packet.end()});
            state = await_state(link);
            check(state.yaw_rad == 1 && state.pitch_rad == -.5f && state.roll_rad == 0 && state.mode == 2 && state.flags == 0,
                  "all split points, little-endian fields and reserved STATE.flags=0");
            check(state.receive_timestamp_ms >= before && state.receive_timestamp_ms <= serial::monotonic_time_ms(),
                  "timestamp after complete packet validation");
            check(!link.receive(state), "sample consumed once");
            link.close();
        }
        reset_transport();
        {
            ::serial link;
            link.open("TEST");
            for (auto byte : packet) feed({byte});
            check(await_state(link).pitch_rad == -.5f, "bytewise reception");
            auto last = packet;
            last[3] = 0xA5; last[15] = 4; last[16] = 0x80;
            last.back() = serial::crc8(last.data() + 1, 16);
            auto stream = packet;
            stream.insert(stream.end(), last.begin(), last.end());
            feed(stream);
            const auto state = await_state(link);
            check(state.mode == 4 && state.flags == 0x80 && state.yaw_rad > 1,
                  "latest coalesced STATE and embedded A5");
            link.send_target(1, -.5f, true);
            link.send_mode(HOST_AUTO_AIM);
            link.close();
        }
        check(closed == 1, "port closed once");
        check(std::find(outgoing.begin(), outgoing.end(), std::vector<std::uint8_t>{
                  0xA5, 1, 9, 0, 0, 0x80, 0x3F, 0, 0, 0, 0xBF, 1, 0x0A}) != outgoing.end(), "TARGET golden frame and close drain");
        check(std::find(outgoing.begin(), outgoing.end(), std::vector<std::uint8_t>{0xA5, 3, 1, 2, 0xA6}) != outgoing.end(),
              "MODE golden frame and close drain");
        check(std::find(outgoing.begin(), outgoing.end(), std::vector<std::uint8_t>{0xA5, 4, 0, 0x54}) != outgoing.end(),
              "HEARTBEAT golden frame");

        reset_transport();
        {
            ::serial link;
            link.open("TEST");
            feed({0xA5, 4});
            feed({0, 0x54});
            feed({0x12, 0x34, 0xA5, 2, 25, 0xA5, 2, 0, 0xA5, 4, 1});
            auto bad = packet;
            bad.back() ^= 1;
            feed(bad);
            auto stream = std::vector<std::uint8_t>{0xA5, 4, 0, 0x54};
            auto reserved = packet;
            reserved[13] = 0x80; reserved[14] = 0x3F;
            reserved.back() = serial::crc8(reserved.data() + 1, 16);
            stream.insert(stream.end(), reserved.begin(), reserved.end());
            stream.insert(stream.end(), {0xA5, 4, 0, 0x54});
            feed(stream);
            const auto state = await_state(link);
            check(state.yaw_rad == 1 && state.roll_rad == 0, "resync, heartbeat and ignored roll");
            link.send_target(1, -.5f, true);
            std::this_thread::sleep_for(std::chrono::milliseconds(220));
            GimbalState extra;
            check(!link.receive(extra), "heartbeat does not publish STATE");
            link.close();
        }
        int valid_targets = 0, invalid_targets = 0;
        for (const auto &frame : outgoing)
        {
            if (frame[1] != HOST_TARGET) continue;
            check(serial::crc8(frame.data() + 1, 11) == frame[12], "periodic TARGET CRC");
            if (frame[11])
            {
                check(invalid_targets == 0, "stale target never becomes valid again");
                ++valid_targets;
            }
            else if (valid_targets) ++invalid_targets;
        }
        check(valid_targets >= 2 && invalid_targets >= 2, "periodic targets expire after vision stalls");

        for (int failure = 2; failure < 4; ++failure)
        {
            reset_transport();
            ::serial link;
            link.open("TEST");
            if (failure < 2)
            {
                auto bad = packet;
                if (failure == 0) bad[0] = 0;
                else bad.back() ^= 1;
                feed(bad);
            }
            else if (failure == 2) fail_read = true;
            else { fail_write = true; link.send_target(0, 0, false); }
            bool failed = false;
            try { await_state(link); } catch (const std::runtime_error &error)
            {
                failed = std::string(error.what()).find("timeout") == std::string::npos || failure == 3;
            }
            check(failed, "worker reports protocol/read/write failure");
            failed = false;
            try { link.close(); } catch (const std::runtime_error &) { failed = true; }
            check(failed && closed == 1, "close reports failure after releasing resources");
            link.close();
            link.open("TEST");
            link.close();
        }
        reset_transport();
        {
            ::serial link;
            link.open("TEST");
            link.send_target(1, -.5f, true);
        }
        check(closed == 1 && std::any_of(outgoing.begin(), outgoing.end(), [](const auto &p) { return p[1] == HOST_TARGET; }),
              "destructor drains queued target");
#endif
        ::serial link;
        GimbalState state;
        check(!link.receive(state), "unopened link has no sample");
        bool rejected = false;
        try { link.open("ARMOR_TEST_PORT_DOES_NOT_EXIST"); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "missing port fails clearly");
        rejected = false;
        try { link.send_target(1, 0, true); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "closed port rejects target send");
        rejected = false;
        try { link.send_mode(HOST_IDLE); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "closed port rejects mode send");
        link.close(); link.close();
        std::cout << "Serial protocol, worker and shutdown checks passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
