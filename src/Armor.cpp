#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "pose_base.hpp"
#include "predictor_single.hpp"
#include "predictor_polar.hpp"
#include "predictor_armor.hpp"
#include "gimbal.hpp"
#include "serial.hpp"
#include "output.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>

CameraSource camera_source = CAMERA;
const int camera_device = 1; // AX USB2.0（外接 OV2710），0 为笔记本内置摄像头
const double camera_recording_fps = 60;
const int video_camera_profile = 2;
const char *const video_file = "assets/video/video_2.avi";
const EnemyColor target_color = ENEMY_RED;
PredictorType predictor_type = PREDICTOR_ARMOR;
bool preview = true;
const double prediction_horizon_ms = 50;
const char *const serial_port = "COM3";

// 临时使用视频 2 的 1280×1024 标定，OV2710 标定完成后替换。
const cv::Mat live_camera_matrix = (cv::Mat_<double>(3, 3) <<
    1711.311186, 0, 732.488057,
    0, 1714.616882, 546.930868,
    0, 0, 1);
const cv::Mat live_distortion = (cv::Mat_<double>(1, 5) <<
    -.119922, -.078593, .007511, -.028028, 0);
const CameraCalibration camera_calibration{
    {0, 0, 0},
    {0, 0, 0},
    Eigen::Quaterniond(1, 0, 0, 0),
    {0, 0, 0},
    {1, 1, 1}
};
const GimbalConfig gimbal_config;

int main()
{
    try
    {
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        ::serial serial;
        if (camera_source == CAMERA)
            serial.open(serial_port);
        ::pose pose_base(camera_source == CAMERA ? camera_calibration : CameraCalibration{});
        if (camera_source == CAMERA)
            pose_base.receive(serial);
        ::camera camera;
        camera.open(camera_source, root / std::filesystem::u8path(video_file), camera_device, camera_recording_fps);
        ::solver solver(camera_source == CAMERA ? live_camera_matrix : cv::Mat(), camera_source == CAMERA ? live_distortion : cv::Mat());
        if (camera_source == VIDEO)
            solver.use_video_profile(video_camera_profile);
        ::predictor_single single_plate;
        ::predictor_polar polar;
        ::predictor_armor armor;
        ::output output;
        output.open(root, camera, predictor_type, preview, prediction_horizon_ms);
        if (camera_source == CAMERA)
            serial.send_mode(HOST_AUTO_AIM);
        do
        {
            const auto frame_start = std::chrono::steady_clock::now();
            auto detections = detector::detect(camera.image_, target_color);
            const auto camera_poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
            if (camera_source == VIDEO)
                pose_base.push({camera.timestamp_ms, 0, 0, 0, HOST_IDLE, 0});
            if (camera_source == CAMERA)
                pose_base.synchronize(serial, camera.timestamp_ms);
            const auto base_poses = pose_base.convert(camera_poses, camera.timestamp_ms);
            PredictionResult prediction;
            if (predictor_type == PREDICTOR_SINGLE_PLATE)
                prediction = single_plate.update_frame(camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            else if (predictor_type == PREDICTOR_POLAR)
                prediction = polar.update_frame(camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            else
                prediction = armor.update_frame(camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            const auto control = gimbal::solve(prediction, pose_base, camera.timestamp_ms,
                camera_source == CAMERA ? serial::monotonic_time_ms() : camera.timestamp_ms, gimbal_config);
            if (camera_source == CAMERA)
                serial.send_target(control.yaw_rad, control.pitch_rad, control.valid());
            if (!output.write(camera, solver, camera_poses, base_poses, pose_base, prediction, control, frame_start))
                break;
            if (camera_source == CAMERA)
                pose_base.receive(serial);
        } while (camera.next());
        if (camera_source == CAMERA)
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
