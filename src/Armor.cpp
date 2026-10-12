#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "pose_base.hpp"
#include "predictor_single.hpp"
#include "predictor_armor.hpp"
#include "gimbal.hpp"
#include "serial.hpp"
#include "output.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

CameraSource camera_source = CAMERA;
const int camera_device = 1;
const double camera_recording_fps = 60; // 录像时间轴帧率，按采集时间戳补帧或丢帧，不代表实际供帧速度
const int video_camera_profile = 1;
const char *const video_file = "assets/video/video_1.avi";
const EnemyColor target_color = ENEMY_RED;
PredictorType predictor_type = PREDICTOR_SINGLE_PLATE;
bool preview = true;
const bool connect_mcu = false;      // 仅摄像头模式：true 使用下位机姿态和基座系，false 直接使用相机系
const bool auto_aim_enabled = false; // 仅摄像头连接下位机时生效
const double prediction_horizon_ms = 100;
const char *const serial_port = "COM3";

// 临时使用视频 2 的 1280×1024 标定，OV2710 标定完成后替换。
const cv::Mat live_camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0, 732.488057,
                                    0, 1714.616882, 546.930868,
                                    0, 0, 1);
const cv::Mat live_distortion = (cv::Mat_<double>(1, 5) << -.119922, -.078593, .007511, -.028028, 0);
// 相机安装位移和朝向尚未标定，以下零偏移、单位旋转用于临时近似。
// 基座原点在 yaw 转轴中心，初始视线的水平投影定义 Z；右转 yaw、抬头 pitch 为正。
const CameraCalibration camera_calibration = []
{
    CameraCalibration calibration;
    calibration.yaw_to_pitch = {0, 0, 0};                         // yaw 转轴到 pitch 转轴的位移，米
    calibration.camera_in_pitch = {0, 0, 0};                      // pitch 转轴到相机光心的位移，米
    calibration.camera_to_pitch = Eigen::Quaterniond::Identity(); // 相机到安装坐标系的旋转
    calibration.angle_zero_rad = {0, 0, 0};                       // yaw、pitch、roll 的零位，弧度
    calibration.angle_direction = {1, 1, 1};                      // yaw、pitch、roll 的正负方向
    return calibration;
}();
const GimbalConfig gimbal_config;

void run_video(const std::filesystem::path &root)
{
    ::camera camera;
    camera.open(VIDEO, root / std::filesystem::u8path(video_file));
    ::solver solver;
    solver.use_video_profile(video_camera_profile);
    ::predictor_single single_plate;
    ::predictor_armor armor;
    ::output output;
    output.open(root, camera, predictor_type, preview, prediction_horizon_ms, false);
    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        auto detections = detector::detect(camera.image, target_color);
        const auto camera_poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
        PredictionResult prediction;
        if (predictor_type == PREDICTOR_SINGLE_PLATE)
            prediction = single_plate.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);
        else
            prediction = armor.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);
        const auto control = gimbal::simulate(prediction, gimbal_config);
        if (!output.write(camera, solver, camera_poses, prediction, control, frame_start))
            break;
    } while (camera.next());
    output.finish();
}

void run_camera(const std::filesystem::path &root)
{
    if (!connect_mcu)
    {
        ::camera camera;
        camera.open(CAMERA, {}, camera_device, camera_recording_fps);
        ::solver solver(live_camera_matrix, live_distortion);
        ::predictor_single single_plate;
        ::predictor_armor armor;
        ::output output;
        output.open(root, camera, predictor_type, preview, prediction_horizon_ms, false);
        do
        {
            const auto frame_start = std::chrono::steady_clock::now();
            auto detections = detector::detect(camera.image, target_color);
            const auto camera_poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
            PredictionResult prediction;
            if (predictor_type == PREDICTOR_SINGLE_PLATE)
                prediction = single_plate.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);
            else
                prediction = armor.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);
            const auto control = gimbal::simulate(prediction, gimbal_config);
            if (!output.write(camera, solver, camera_poses, prediction, control, frame_start))
                break;
        } while (camera.next());
        output.finish();
        return;
    }

    ::serial serial;
    ::pose pose_base(camera_calibration);
    GimbalState gimbal_state;
    serial.open(serial_port);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (!serial.receive(gimbal_state))
    {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("No initial gimbal state received");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pose_base.update(gimbal_state);
    ::camera camera;
    camera.open(CAMERA, {}, camera_device, camera_recording_fps);
    ::solver solver(live_camera_matrix, live_distortion);
    ::predictor_single single_plate;
    ::predictor_armor armor;
    ::output output;
    output.open(root, camera, predictor_type, preview, prediction_horizon_ms, true);
    serial.send_mode(auto_aim_enabled ? HOST_AUTO_AIM : HOST_IDLE);
    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        if (serial.receive(gimbal_state))
            pose_base.update(gimbal_state);
        auto detections = detector::detect(camera.image, target_color);
        const auto camera_poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
        const auto base_poses = pose_base.convert(camera_poses);
        PredictionResult prediction;
        if (predictor_type == PREDICTOR_SINGLE_PLATE)
            prediction = single_plate.update_frame(camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
        else
            prediction = armor.update_frame(camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);

        ControlTarget control;
        if (auto_aim_enabled)
            control = gimbal::solve(prediction, pose_base, camera.timestamp_ms,
                                    serial::monotonic_time_ms(), gimbal_config);
        else
        {
            control.yaw_rad = pose_base.latest_state_.yaw_rad;
            control.pitch_rad = pose_base.latest_state_.pitch_rad;
            control.status = "disabled";
        }
        serial.send_target(control.yaw_rad, control.pitch_rad, control.valid());
        if (!output.write(camera, solver, camera_poses, prediction, control, frame_start, &base_poses, &pose_base))
            break;
    } while (camera.next());

    serial.send_target(pose_base.latest_state_.yaw_rad, pose_base.latest_state_.pitch_rad, false);
    serial.send_mode(HOST_IDLE);
    serial.close();
    output.finish();
}

int main()
{
    try
    {
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        if (camera_source == VIDEO)
            run_video(root);
        else
            run_camera(root);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Armor failed: " << error.what() << '\n';
        return 1;
    }
}
