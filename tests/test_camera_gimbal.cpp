#include "pose_base.hpp"
#include "predictor_armor.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

static SolvedFrame camera_frame(const Eigen::Vector3d &position, const Eigen::Vector3d &rvec)
{
    ::armor armor;
    armor.tvec = (cv::Mat_<double>(3, 1) << position.x(), position.y(), position.z());
    armor.rvec = (cv::Mat_<double>(3, 1) << rvec.x(), rvec.y(), rvec.z());
    SolvedFrame frame;
    frame.armors.push_back(std::move(armor));
    return frame;
}

static Eigen::Vector3d position_of(const ::armor &armor)
{
    return {armor.tvec.at<double>(0), armor.tvec.at<double>(1), armor.tvec.at<double>(2)};
}

static Eigen::Matrix3d rotation_of(const ::armor &armor)
{
    cv::Mat rotation;
    cv::Rodrigues(armor.rvec, rotation);
    return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation.ptr<double>());
}

int main()
{
    try
    {
        CameraCalibration mount;
        mount.yaw_to_pitch = {.1, 0, 0};
        mount.camera_in_pitch = {0, 0, .2};
        ::pose frames(mount);
        frames.update({0, 0, 0, 0, HOST_IDLE, 0});
        check((frames.base_from_camera_ * Eigen::Vector3d::UnitZ() - Eigen::Vector3d(.1, 0, 1.2)).norm() < 1e-12,
              "lever arm relative to yaw-axis origin");
        check((frames.base_from_camera_.translation() - Eigen::Vector3d(.1, 0, .2)).norm() < 1e-12,
              "camera mounting offset is retained at startup");
        frames.update({100, static_cast<float>(CV_PI / 2), 0, 0, HOST_IDLE, 0});
        check((frames.base_from_camera_ * Eigen::Vector3d::UnitZ() - Eigen::Vector3d(1.2, 0, -.1)).norm() < 1e-6,
              "base yaw rotation");
        frames.update({200, static_cast<float>(CV_PI / 2), static_cast<float>(CV_PI / 2), 0, HOST_IDLE, 0});
        const auto &T = frames.base_from_camera_;
        check((T * Eigen::Vector3d::UnitZ() - Eigen::Vector3d(0, -1.2, -.1)).norm() < 1e-6, "yaw-pitch order");
        const Eigen::Vector3d point(.4, -.3, 2);
        check((T.inverse() * (T * point) - point).norm() < 1e-12, "round trip");

        ::pose tilted(mount);
        tilted.update({0, static_cast<float>(CV_PI / 3), static_cast<float>(CV_PI / 6), 0, HOST_IDLE, 0});
        const Eigen::Vector3d initial_forward = tilted.base_from_camera_.linear().col(2);
        check((initial_forward - Eigen::Vector3d(0, -.5, std::sqrt(3.) / 2)).norm() < 1e-6,
              "startup heading is removed while pitch remains relative to the ground");
        check((tilted.base_from_camera_.translation() - Eigen::Vector3d(.1, -.1, .1 * std::sqrt(3.))).norm() < 1e-6,
              "initial pitch rotates the mounting offset without moving the base origin");
        tilted.update({100, static_cast<float>(CV_PI / 2), static_cast<float>(CV_PI / 6), 0, HOST_IDLE, 0});
        const Eigen::Vector3d turned_forward = tilted.base_from_camera_.linear().col(2);
        check(turned_forward.x() > 0 && std::abs(turned_forward.y() + .5) < 1e-6,
              "positive yaw turns right and positive pitch points up");

        CameraCalibration yaw_mount;
        yaw_mount.camera_to_pitch = Eigen::AngleAxisd(.25, Eigen::Vector3d::UnitY());
        ::pose mounted(yaw_mount);
        mounted.update({0, static_cast<float>(40 * CV_PI / 180), static_cast<float>(20 * CV_PI / 180), 0, HOST_IDLE, 0});
        const Eigen::Vector3d mounted_forward = mounted.base_from_camera_.linear().col(2);
        check(std::abs(mounted_forward.x()) < 1e-12 && mounted_forward.z() > 0 && mounted_forward.y() < 0,
              "initial heading follows the camera optical axis including mounting rotation");
        bool vertical_rejected = false;
        try
        {
            ::pose vertical;
            vertical.update({0, 0, static_cast<float>(CV_PI / 2), 0, HOST_IDLE, 0});
        }
        catch (const std::invalid_argument &) { vertical_rejected = true; }
        check(vertical_rejected, "vertical initial view cannot define a horizontal forward axis");

        ::pose empty;
        bool empty_rejected = false;
        try { empty.convert({}); }
        catch (const std::logic_error &) { empty_rejected = true; }
        check(empty_rejected, "conversion requires an initial attitude");
        empty_rejected = false;
        try { empty.at_angles(Eigen::Vector3d::Zero()); }
        catch (const std::logic_error &) { empty_rejected = true; }
        check(empty_rejected, "absolute-angle control requires an initial heading");
        empty.update({0, 0, 0, 0, HOST_IDLE, 0});
        const auto empty_frame = empty.convert({});
        check(empty_frame.armors.empty() && empty_frame.observations.empty(), "empty detections stay empty");

        ::pose crossing;
        crossing.update({0, static_cast<float>(179 * CV_PI / 180), 0, 0, HOST_IDLE, 0});
        crossing.update({100, static_cast<float>(-179 * CV_PI / 180), 0, 0, HOST_IDLE, 0});
        const Eigen::Vector3d crossing_forward(std::sin(2 * CV_PI / 180), 0, std::cos(2 * CV_PI / 180));
        check((crossing.base_from_camera_.linear().col(2) - crossing_forward).norm() < 1e-6,
              "latest yaw across pi produces a two-degree relative turn");

        CameraCalibration corrected_mount;
        corrected_mount.angle_zero_rad = {.25, .5, 0};
        corrected_mount.angle_direction = {-1, 1, 1};
        corrected_mount.camera_to_pitch = Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitZ());
        ::pose corrected(corrected_mount);
        corrected.update({0, 1, .5f, 0, HOST_IDLE, 0});
        corrected.update({20, 1.25f, .6f, 0, HOST_IDLE, 0});
        const Eigen::Matrix3d corrected_rotation =
            (Eigen::AngleAxisd(-.25, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(.1, Eigen::Vector3d::UnitX()) *
             Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        check((corrected.base_from_camera_.linear() - corrected_rotation).norm() < 1e-7,
              "explicit zero, direction and camera mounting rotation");

        ::pose nonzero_heading;
        nonzero_heading.update({10, .8f, .3f, 0, HOST_IDLE, 0});
        for (int i = 0; i < 300; ++i)
            nonzero_heading.update({30 + i * 20., 1.f, .3f, 0, HOST_IDLE, 0});
        check(nonzero_heading.reference_time_ms_ == 10 && std::abs(nonzero_heading.reference_yaw_rad_ - .8) < 1e-6,
              "repeated updates preserve the initial heading");
        check(nonzero_heading.latest_state_.receive_timestamp_ms == 6010, "only the latest feedback is retained");
        const Eigen::Vector3d latest_forward = nonzero_heading.base_from_camera_.linear().col(2);
        check(std::abs(std::atan2(latest_forward.x(), latest_forward.z()) - .2) < 1e-6,
              "later yaw is measured from the original startup heading");
        check((nonzero_heading.base_from_camera_.matrix() - nonzero_heading.at_angles({1, .3f, 0}).matrix()).norm() < 1e-12,
              "absolute-angle control and conversion share the same heading reference");
        const Eigen::Matrix4d unchanged = nonzero_heading.base_from_camera_.matrix();
        nonzero_heading.at_angles({.5, -.2, .1});
        check((nonzero_heading.base_from_camera_.matrix() - unchanged).norm() == 0 &&
              nonzero_heading.latest_state_.yaw_rad == 1.f, "candidate control angles do not mutate current attitude");

        const GimbalState received{serial::monotonic_time_ms(), 1, -.5f, 0, HOST_AUTO_AIM, 0};
        ::pose received_frames;
        received_frames.update(received);
        const auto received_pose = received_frames.convert(camera_frame({0, 0, 2}, Eigen::Vector3d::Zero()));
        const Eigen::Vector3d received_position(0, 2 * std::sin(.5), 2 * std::cos(.5));
        const Eigen::Vector4d received_measurement(0, .5, 2, 0);
        check((position_of(received_pose.armors[0]) - received_position).norm() < 1e-12 &&
              (received_pose.observations[0] - received_measurement).norm() < 1e-12,
              "wire STATE flags=0 produces base position and all four EKF observations");

        CameraCalibration moving_mount;
        moving_mount.yaw_to_pitch = {.1, .01, 0};
        moving_mount.camera_in_pitch = {.03, -.015, .15};
        moving_mount.camera_to_pitch = Eigen::AngleAxisd(.08, Eigen::Vector3d::UnitX()) *
                                        Eigen::AngleAxisd(-.12, Eigen::Vector3d::UnitZ());
        ::pose moving(moving_mount);
        const Eigen::Vector3d fixed_position(.2, -.1, 3);
        const Eigen::Matrix3d fixed_rotation = (Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(.12, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(-.15, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        const cv::Mat camera = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
        ::solver solver(camera, distortion);
        const std::vector<cv::Point3f> object_points = {{-.0675f, -.028f, 0}, {-.0675f, .028f, 0},
                                                       {.0675f, .028f, 0}, {.0675f, -.028f, 0}};
        for (int i = 0; i < 5; ++i)
        {
            moving.update({i * 20., i * .05f, i * -.025f, i * .01f, HOST_IDLE, 0});
            const auto camera_from_base = moving.base_from_camera_.inverse();
            const Eigen::Vector3d camera_position = camera_from_base * fixed_position;
            const Eigen::Matrix3d camera_rotation = camera_from_base.linear() * fixed_rotation;
            const Eigen::AngleAxisd camera_angle_axis(camera_rotation);
            const Eigen::Vector3d camera_rvec = camera_angle_axis.axis() * camera_angle_axis.angle();
            const auto exact = moving.convert(camera_frame(camera_position, camera_rvec));
            check((position_of(exact.armors[0]) - fixed_position).norm() < 1e-12 &&
                  (rotation_of(exact.armors[0]) - fixed_rotation).norm() < 1e-12,
                  "full position and rotation remain fixed under yaw/pitch/roll camera motion");
            const auto projected_frame = camera_frame(camera_position, camera_rvec);
            std::vector<cv::Point2f> pixels;
            cv::projectPoints(object_points, projected_frame.armors[0].rvec, projected_frame.armors[0].tvec,
                              camera, distortion, pixels);
            ::armor detected;
            for (int j = 0; j < 4; ++j) detected.vertices[j] = pixels[j];
            check(solver.solve(detected), "moving-camera synthetic armor PnP solved");
            SolvedFrame solved_frame;
            solved_frame.armors.push_back(detected);
            const auto solved = moving.convert(solved_frame);
            check((position_of(solved.armors[0]) - fixed_position).norm() < .003 &&
                  (rotation_of(solved.armors[0]) - fixed_rotation).norm() < .01,
                  "PnP and lower-controller attitude recover a stationary armor pose");
        }

        ::pose turned;
        turned.update({0, 0, 0, 0, HOST_IDLE, 0});
        turned.update({20, 2, 0, 0, HOST_IDLE, 0});
        const auto back_pose = turned.convert(camera_frame({0, 0, 3}, Eigen::Vector3d::Zero()));
        ::predictor_armor fixed;
        check(fixed.update_frame(0, back_pose.observations).status == "initialized",
              "fixed-frame Armor accepts a target behind the base Z axis");
        PredictionResult simulated;
        check(!gimbal::simulate(simulated).valid(), "simulation without target");
        simulated.status = "updated";
        simulated.future.plates.emplace_back(.3, -.2, 3, 0);
        const auto command = gimbal::simulate(simulated);
        check(command.valid() && command.armor_id == 0, "simulation target selection");
        check(std::abs(command.yaw_rad - std::atan2(.3, 3)) < 1e-6 &&
              std::abs(command.pitch_rad - std::atan2(.2, std::hypot(.3, 3))) < 1e-6,
              "simulation camera angles");
        simulated.future.plates[0](0) = 10;
        check(gimbal::simulate(simulated).status == "out_of_limits", "simulation angle limits");
        std::cout << "PoseBase checks passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
