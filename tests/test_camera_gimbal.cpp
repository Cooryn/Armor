#include "pose_base.hpp"
#include "predictor_armor.hpp"
#include "serial.hpp"
#include "solver.hpp"
#include <iostream>

static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        CameraCalibration mount;
        mount.yaw_to_pitch = {0.1, 0, 0};
        mount.camera_in_pitch = {0, 0, 0.2};
        PoseBase frames{mount, {{0, 0, 0, 0}, {100, 90, 0, 0}, {200, 90, 90, 0}}};
        frames.reset_origin(0);
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
        check(!frames.covered(300), "origin reset requires covered timestamp");
        PoseBase interpolated{mount, {{0, 0, 0, 0}, {100, 90, 0, 0}}, 100};
        interpolated.reset_origin(50);
        check(interpolated.at(50).translation().norm() < 1e-12, "interpolated reference origin");
        PoseBase crossing{{}, {{0, 179, 0, 0}, {100, -179, 0, 0}}};
        check((crossing.at(50).linear() * Eigen::Vector3d::UnitZ() + Eigen::Vector3d::UnitZ()).norm() < 1e-12,
              "shortest interpolation across wrap");
        check(!crossing.covered(101), "no telemetry extrapolation");
        PoseBase sparse{{}, {{0, 0, 0, 0}, {500, 0, 0, 0}}};
        check(!sparse.covered(250), "telemetry gap is unavailable");

        PoseBase live{mount, {}};
        live.push({10, 0, 0, 0, HOST_IDLE, 0});
        live.push({110, static_cast<float>(CV_PI / 2), 0, 0, HOST_IDLE, 0});
        const Eigen::Vector3d halfway(1.3 / std::sqrt(2.) - .1, 0, 1.1 / std::sqrt(2.) - .2);
        check((live.at(60) * Eigen::Vector3d::UnitZ() - halfway).norm() < 1e-6,
              "serial radians converted/interpolated with lever arms");
        check(live.reference_time_ms_ == 10 && (live.origin_in_mechanical_ - Eigen::Vector3d(.1, 0, .2)).norm() < 1e-12,
              "first serial sample fixes the optical-center origin");
        check(!live.covered(111), "live attitude requires received coverage");
        PoseBase serial_crossing{};
        serial_crossing.push({0, static_cast<float>(179 * CV_PI / 180), 0, 0, HOST_IDLE, 0});
        serial_crossing.push({20, static_cast<float>(-179 * CV_PI / 180), 0, 0, HOST_IDLE, 0});
        check((serial_crossing.at(10).linear() * Eigen::Vector3d::UnitZ() + Eigen::Vector3d::UnitZ()).norm() < 1e-6,
              "serial angle wrap interpolates through pi");
        PoseBase dropped{};
        dropped.push({0, 0, 0, 0, HOST_IDLE, 0});
        dropped.push({200, .2f, 0, 0, HOST_IDLE, 0});
        check(!dropped.covered(100), "missing serial coverage");
        CameraCalibration corrected_mount;
        corrected_mount.angle_zero_rad = {.25, .5, 0};
        corrected_mount.angle_direction = {-1, 1, 1};
        corrected_mount.camera_to_pitch = Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitZ());
        PoseBase corrected{corrected_mount, {}};
        corrected.push({0, 1, .5f, 0, HOST_IDLE, 0});
        const Eigen::Matrix3d corrected_rotation =
            (Eigen::AngleAxisd(-.75, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        check((corrected.at(0).linear() - corrected_rotation).norm() < 1e-12, "explicit zero, direction and camera mounting rotation");
        for (int i = 0; i < 300; ++i)
            live.push({130 + i * 20., 0, 0, 0, HOST_IDLE, 0});
        check(live.samples_.size() == 256 && live.reference_time_ms_ == 10 &&
              (live.origin_in_mechanical_ - Eigen::Vector3d(.1, 0, .2)).norm() < 1e-12,
              "bounded live cache preserves the fixed origin when old samples are removed");

        // The serial regression verifies decoding; PoseBase consumes this typed STATE.
        const GimbalState received{monotonic_time_ms(), 1, -.5f, 0, HOST_AUTO_AIM, 0};
        PoseBase received_frames{};
        received_frames.push(received);
        const auto received_pose = received_frames.to_base({0, 0, 2}, Eigen::Vector3d::Zero(), received.receive_timestamp_ms);
        const Eigen::Vector3d received_position(2 * std::sin(1.) * std::cos(.5), 2 * std::sin(.5), 2 * std::cos(1.) * std::cos(.5));
        check((received_pose.position - received_position).norm() < 1e-12 &&
              std::abs(received_pose.measurement(0) - 1) < 1e-12 &&
              std::abs(received_pose.measurement(1) - .5) < 1e-12 &&
              std::abs(received_pose.measurement(2) - 2) < 1e-12 &&
              std::abs(received_pose.measurement(3) + 1) < 1e-12,
              "wire STATE flags=0 produces base position and all four EKF observations");

        CameraCalibration moving_mount;
        moving_mount.yaw_to_pitch = {.1, .01, 0};
        moving_mount.camera_in_pitch = {.03, -.015, .15};
        moving_mount.camera_to_pitch = Eigen::AngleAxisd(.08, Eigen::Vector3d::UnitX()) *
                                        Eigen::AngleAxisd(-.12, Eigen::Vector3d::UnitZ());
        PoseBase moving{moving_mount, {}};
        for (int i = 0; i < 5; ++i)
            moving.push({i * 20., i * .05f, i * -.025f, i * .01f, HOST_IDLE, 0});
        const Eigen::Vector3d fixed_position(.2, -.1, 3);
        const Eigen::Matrix3d fixed_rotation = (Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(.12, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(-.15, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        const cv::Mat camera = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
        Solver pnp(camera, distortion);
        const std::vector<cv::Point3f> object_points = {{-.0675f, -.028f, 0}, {-.0675f, .028f, 0},
                                                       {.0675f, .028f, 0}, {.0675f, -.028f, 0}};
        for (double timestamp : {0., 10., 30., 50., 80.})
        {
            const auto camera_from_base = moving.at(timestamp).inverse();
            const Eigen::Vector3d camera_position = camera_from_base * fixed_position;
            const Eigen::Matrix3d camera_rotation = camera_from_base.linear() * fixed_rotation;
            const Eigen::AngleAxisd camera_angle_axis(camera_rotation);
            const Eigen::Vector3d camera_rvec = camera_angle_axis.axis() * camera_angle_axis.angle();
            const auto exact = moving.to_base(camera_position, camera_rvec, timestamp);
            const Eigen::Matrix3d restored_rotation = Eigen::AngleAxisd(exact.rvec.norm(), exact.rvec.normalized()).toRotationMatrix();
            check((exact.position - fixed_position).norm() < 1e-12 && (restored_rotation - fixed_rotation).norm() < 1e-12,
                  "full position and rotation remain fixed under yaw/pitch/roll camera motion");
            cv::Mat rvec = (cv::Mat_<double>(3, 1) << camera_rvec.x(), camera_rvec.y(), camera_rvec.z());
            cv::Mat tvec = (cv::Mat_<double>(3, 1) << camera_position.x(), camera_position.y(), camera_position.z());
            std::vector<cv::Point2f> pixels;
            cv::projectPoints(object_points, rvec, tvec, camera, distortion, pixels);
            Armor detected;
            for (int i = 0; i < 4; ++i) detected.vertices[i] = pixels[i];
            check(pnp.solve(detected), "moving-camera synthetic armor PnP solved");
            const auto solved = moving.to_base({detected.tvec.at<double>(0), detected.tvec.at<double>(1), detected.tvec.at<double>(2)},
                {detected.rvec.at<double>(0), detected.rvec.at<double>(1), detected.rvec.at<double>(2)}, timestamp);
            const Eigen::Matrix3d solved_rotation = Eigen::AngleAxisd(solved.rvec.norm(), solved.rvec.normalized()).toRotationMatrix();
            check((solved.position - fixed_position).norm() < .003 && (solved_rotation - fixed_rotation).norm() < .01,
                  "PnP and lower-controller attitude recover a stationary armor pose");
        }
        check(moving.covered(10) && !moving.covered(100) && !dropped.covered(100), "synchronization availability");
        PoseBase turned;
        turned.push({0, 2, 0, 0, HOST_IDLE, 0});
        const auto back_pose = turned.to_base({0, 0, 3}, Eigen::Vector3d::Zero(), 0);
        ArmorEKF fixed;
        fixed.base_frame = true;
        check(fixed.update_frame(0, 0, {{back_pose.position, back_pose.measurement}}).initialized,
              "fixed-frame Armor accepts a target behind the base Z axis");
        std::cout << "PoseBase checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
