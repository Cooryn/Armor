#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "pose_base.hpp"
#include "predictor.hpp"
#include "predictor_polar.hpp"
#include "predictor_armor.hpp"
#include "gimbal.hpp"
#include "serial.hpp"
#include "output.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>

// Edit this configuration and rebuild. Video paths are relative to the project root.
constexpr CameraSource camera_source = CAMERA_VIDEO;
constexpr int camera_device = 0;
constexpr double camera_recording_fps = 30;
constexpr int video_camera_profile = 2; // Select calibrated intrinsics: video 1 or video 2.
constexpr const char *video_file = "assets/video/video_2.avi";
constexpr EnemyColor target_color = ENEMY_RED;
constexpr PredictorType predictor_type = PREDICTOR_ARMOR;
constexpr bool preview = true;
constexpr double prediction_horizon_ms = 50;
constexpr const char *serial_port = "COM3";

// Fill the real camera/gimbal calibration before enabling live control.
const cv::Mat live_camera_matrix; // 3x3 CV_64F camera intrinsics.
const cv::Mat live_distortion;    // CV_64F distortion coefficients (zero for an ideal camera).
const CameraCalibration camera_calibration{
    {0, 0, 0}, // yaw axis -> pitch axis, m
    {0, 0, 0}, // pitch axis -> camera optical center, m
    Eigen::Quaterniond(1, 0, 0, 0), // camera -> pitch mount rotation
    {0, 0, 0}, // raw yaw/pitch/roll zero, rad
    {1, 1, 1}  // raw angle directions, each +1 or -1
};
const GimbalConfig gimbal_config; // Absolute raw-angle limits and age/uncertainty gates.

int main()
{
    try
    {
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        constexpr bool live = camera_source == CAMERA_OPENCV;
        Serial serial;
        if constexpr (live)
            serial.open(serial_port);
        PoseBase pose_base(live ? camera_calibration : CameraCalibration{});
        if constexpr (live)
            pose_base.receive(serial);
        Camera camera;
        camera.open({camera_source, root / std::filesystem::u8path(video_file), camera_device, camera_recording_fps});
        Solver pnp(live ? live_camera_matrix : cv::Mat(), live ? live_distortion : cv::Mat());
        if constexpr (!live)
            pnp.use_video_profile(video_camera_profile);
        SinglePlateEKF single_plate;
        PolarEKF polar;
        ArmorEKF armor;
        armor.base_frame = true;
        Gimbal gimbal(gimbal_config);
        Output output;
        output.open(root, camera, predictor_type, preview);
        if constexpr (live)
            serial.send_mode(HOST_AUTO_AIM);
        do
        {
            const auto frame_start = std::chrono::steady_clock::now();
            auto detections = detectArmors(camera.image_, target_color);
            const auto camera_poses = pnp.solve_frame(std::move(detections), camera.timestamp_ms);
            if (!live)
                pose_base.push({camera.timestamp_ms, 0, 0, 0, HOST_IDLE, 0}); // Fixed virtual camera for replay.
            if constexpr (live)
                pose_base.synchronize(serial, camera.timestamp_ms);
            const auto base_poses = pose_base.convert(camera_poses, camera.timestamp_ms);
            VideoPrediction prediction;
            if constexpr (predictor_type == PREDICTOR_SINGLE_PLATE)
                prediction = single_plate.update_frame(camera.frame_id_, camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            else if constexpr (predictor_type == PREDICTOR_POLAR)
                prediction = polar.update_frame(camera.frame_id_, camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            else
                prediction = armor.update_frame(camera.frame_id_, camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            const auto command = gimbal.solve(prediction, pose_base, camera.timestamp_ms,
                live ? monotonic_time_ms() : camera.timestamp_ms);
            if constexpr (live)
                serial.send_target(command.yaw_rad, command.pitch_rad, command.valid);
            if (!output.write(camera, pnp, camera_poses, base_poses, pose_base, prediction, command, frame_start))
                break;
            if constexpr (live)
                pose_base.receive(serial); // Keep a left-hand pose sample before acquiring the next frame.
        } while (camera.next());
        if constexpr (live)
        {
            serial.send_target(pose_base.latest_state_.yaw_rad, pose_base.latest_state_.pitch_rad, false);
            serial.send_mode(HOST_IDLE);
            serial.close();
        }
        output.finish();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Armor failed: " << error.what() << '\n';
        return 1;
    }
}
