#include "camera_gimbal.hpp"
#include "camera_frames.hpp"
#include <iostream>

static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        CameraFrames::Calibration mount;
        mount.yaw_to_pitch = {0.1, 0, 0};
        mount.camera_in_pitch = {0, 0, 0.2};
        CameraFrames frames(mount, {{0, 0, 0, 0}, {100, 90, 0, 0}, {200, 90, 90, 0}});
        auto T = frames.at(0);
        check((T * Eigen::Vector3d(0, 0, 1) - Eigen::Vector3d(0, 0, 1)).norm() < 1e-12, "lever arm");
        T = frames.at(100);
        check((T * Eigen::Vector3d(0, 0, 1) - Eigen::Vector3d(1.1, 0, -.3)).norm() < 1e-12, "base yaw rotation");
        T = frames.at(200);
        check((T * Eigen::Vector3d(0, 0, 1) - Eigen::Vector3d(-.1, -1.2, -.3)).norm() < 1e-12, "yaw-pitch order");
        const Eigen::Vector3d point(.4, -.3, 2);
        check((T.inverse() * (T * point) - point).norm() < 1e-12, "round trip");
        check(frames.at(0).translation().norm() < 1e-12, "reference optical center is origin");
        const auto new_from_old = frames.reset_origin(100);
        check(frames.at(100).translation().norm() < 1e-12, "reset optical center is origin");
        check((frames.at(200).matrix() - (new_from_old * T).matrix()).norm() < 1e-12, "reset maps old frame to new");
        check((frames.at(200).linear() - T.linear()).norm() < 1e-12, "reset preserves base axes");
        check((frames.reset_origin(100).matrix() - Eigen::Matrix4d::Identity()).norm() < 1e-12, "repeated reset identity");
        bool reset_rejected = false;
        try { frames.reset_origin(300); } catch (const std::invalid_argument &) { reset_rejected = true; }
        check(reset_rejected && frames.reference_time_ms() == 100, "failed reset preserves origin");
        CameraFrames interpolated(mount, {{0, 0, 0, 0}, {100, 90, 0, 0}}, 100, 50);
        check(interpolated.at(50).translation().norm() < 1e-12, "interpolated reference origin");
        CameraFrames crossing({}, {{0, 179, 0, 0}, {100, -179, 0, 0}});
        check((crossing.at(50).linear() * Eigen::Vector3d::UnitZ() + Eigen::Vector3d::UnitZ()).norm() < 1e-12,
              "shortest interpolation across wrap");
        check((crossing.at(100 + 1e-9).matrix() - crossing.at(100).matrix()).norm() == 0,
              "CSV time rounding at boundary");
        bool rejected = false;
        try { crossing.at(101); } catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "no telemetry extrapolation");
        CameraFrames sparse({}, {{0, 0, 0, 0}, {500, 0, 0, 0}});
        rejected = false;
        try { sparse.at(250); } catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "telemetry gap rejected");
        CameraGimbal controller;
        auto out = controller.update(0, {1, -1, 1}, true);
        check(out.target_valid && !out.control_valid && out.status == "initializing", "initialization");
        out = controller.update(100, {1, -1, 1}, true);
        check(std::abs(out.yaw_error_deg - 45) < 1e-10 && out.pitch_error_deg > 0, "angle signs");
        check(out.yaw_rate_dps == 18 && out.pitch_rate_dps == 18, "acceleration limit in seconds");
        check(out.speed_limited && out.angle_limited && out.acceleration_limited, "limit flags");
        out = controller.update(200, {1, -1, 1}, false);
        check(!out.control_valid && out.yaw_rate_dps == 0 && out.pitch_rate_dps == 0, "loss stops immediately");
        out = controller.update(300, {0, 0, 2}, true);
        check(out.control_valid && out.yaw_rate_dps == 0 && out.pitch_rate_dps == 0, "center deadband");
        out = controller.update(600, {1, 0, 2}, true);
        check(!out.control_valid && out.status == "time_gap" && out.yaw_rate_dps == 0, "large time gap");
        out = controller.update(700, {0, 0, -1}, true);
        check(!out.target_valid && out.status == "invalid_target", "behind camera");
        out = controller.update(800, {std::numeric_limits<double>::quiet_NaN(), 0, 2}, true);
        check(!out.control_valid && out.yaw_rate_dps == 0, "invalid position");
        bool threw = false;
        try { controller.update(800, {0, 0, 2}, true); }
        catch (const std::invalid_argument &) { threw = true; }
        check(threw, "duplicate timestamp rejected");
        CameraGimbal::Config config;
        config.gain = 0;
        threw = false;
        try { CameraGimbal invalid(config); }
        catch (const std::invalid_argument &) { threw = true; }
        check(threw, "invalid config rejected");
        CameraGimbal reverse;
        reverse.update(0, {-1, 1, 2}, true);
        out = reverse.update(100, {-1, 1, 2}, true);
        check(out.yaw_rate_dps < 0 && out.pitch_rate_dps < 0, "left/down signs");
        std::cout << "Camera tracking control checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
