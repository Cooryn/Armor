#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "pose_base.hpp"
#include "predictor_single.hpp"
#include "predictor_armor.hpp"
#include "gimbal.hpp"
#include "serial.hpp"
#include "log_output.hpp"
#include "video_output.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

CameraSource camera_source = VIDEO;
const int camera_device = 1;
const double camera_recording_fps = 30;
const int video_camera_profile = 2;
const char *const video_file = "assets/video/video_2.avi";

const EnemyColor target_color = ENEMY_RED;

PredictorType predictor_type = PREDICTOR_ARMOR;

const bool connect_mcu = false;      // 仅摄像头模式：true 使用下位机姿态和基座系，false 直接使用相机系
const bool auto_aim_enabled = false; // 仅摄像头连接下位机时生效
const double prediction_horizon_ms = 100;
const char *const serial_port = "COM3";

// OV2710实测标定1280×720
const cv::Mat live_camera_matrix = (cv::Mat_<double>(3, 3) << 724.1270087027805, 0, 634.8734753435976,
                                    0, 724.7744702772803, 361.56645765182736,
                                    0, 0, 1);
const cv::Mat live_distortion = (cv::Mat_<double>(1, 5) << 0.19096982316205274, -0.5174000457589348,
                                 -0.0016604452786579244, -0.0009416271376343901, 0.5296904922932862);

// 相机安装位移和朝向尚未标定，以下用于临时近似
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

// 视频模式
void run_video(const std::filesystem::path &root)
{
    ::camera camera;
    camera.open(VIDEO, root / std::filesystem::u8path(video_file));
    ::solver solver;
    solver.use_video_profile(video_camera_profile);
    ::predictor_single single_plate;
    ::predictor_armor armor;
    ::log_output output;
    output.open(root, camera, predictor_type, prediction_horizon_ms, false);
    video_output recording(log_output::result_directory(root, camera, predictor_type, false), camera);
    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        auto detections = detector::detect(camera.image, target_color);                           // 检测
        const auto camera_poses = solver.solve_frame(std::move(detections), camera.timestamp_ms); // PnP

        // EKF
        PredictionResult prediction;
        if (predictor_type == PREDICTOR_SINGLE_PLATE)
            prediction = single_plate.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);
        else
            prediction = armor.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);

        const auto control = gimbal::simulate(prediction, gimbal_config); // 控制
        output.write(camera, camera_poses, prediction, control);
        const bool running = recording.write(camera, solver, camera_poses, frame_start, &prediction, predictor_type, &control);
        if (!running)
            break;
    } while (camera.next());
    recording.finish();
    output.finish();
}

// 摄像头模式
void run_camera(const std::filesystem::path &root)
{
    if (!connect_mcu)
    {
        ::camera camera;
        camera.open(CAMERA, {}, camera_device, camera_recording_fps);
        ::solver solver(live_camera_matrix, live_distortion);
        ::predictor_single single_plate;
        ::predictor_armor armor;
        ::log_output output;
        output.open(root, camera, predictor_type, prediction_horizon_ms, false);
        video_output recording(log_output::result_directory(root, camera, predictor_type, false), camera);
        do
        {
            const auto frame_start = std::chrono::steady_clock::now();
            auto detections = detector::detect(camera.image, target_color);                           // 检测
            const auto camera_poses = solver.solve_frame(std::move(detections), camera.timestamp_ms); // PnP

            // EKF
            PredictionResult prediction;
            if (predictor_type == PREDICTOR_SINGLE_PLATE)
                prediction = single_plate.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);
            else
                prediction = armor.update_frame(camera.timestamp_ms, camera_poses.observations, prediction_horizon_ms);

            const auto control = gimbal::simulate(prediction, gimbal_config); // 控制
            output.write(camera, camera_poses, prediction, control);
            const bool running = recording.write(camera, solver, camera_poses, frame_start, &prediction, predictor_type, &control);
            if (!running)
                break;
        } while (camera.next());
        recording.finish();
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
    ::log_output output;
    output.open(root, camera, predictor_type, prediction_horizon_ms, true);
    video_output recording(log_output::result_directory(root, camera, predictor_type, true), camera);
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
        output.write(camera, camera_poses, prediction, control, &base_poses, &pose_base);
        const bool running = recording.write(camera, solver, camera_poses, frame_start, &prediction, predictor_type, &control, &pose_base);
        if (!running)
            break;
    } while (camera.next());

    serial.send_target(pose_base.latest_state_.yaw_rad, pose_base.latest_state_.pitch_rad, false);
    serial.send_mode(HOST_IDLE);
    serial.close();
    recording.finish();
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
