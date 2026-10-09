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

CameraSource camera_source = VIDEO;
const int camera_device = 0;
const double camera_recording_fps = 30;
const int video_camera_profile = 2;
const char *const video_file = "assets/video/video_2.avi";
const EnemyColor target_color = ENEMY_RED;
PredictorType predictor_type = PREDICTOR_ARMOR;
bool preview = true;
const double prediction_horizon_ms = 50;
const char *const serial_port = "COM3";

const cv::Mat live_camera_matrix;
const cv::Mat live_distortion;
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
        Serial serial;
        if (camera_source == CAMERA)
            serial.open(serial_port);
        PoseBase pose_base(camera_source == CAMERA ? camera_calibration : CameraCalibration{});
        if (camera_source == CAMERA)
            pose_base.receive(serial);
        Camera camera;
        camera.open({camera_source, root / std::filesystem::u8path(video_file), camera_device, camera_recording_fps});
        Solver pnp(camera_source == CAMERA ? live_camera_matrix : cv::Mat(), camera_source == CAMERA ? live_distortion : cv::Mat());
        if (camera_source == VIDEO)
            pnp.use_video_profile(video_camera_profile);
        SinglePlateEKF single_plate;
        PolarEKF polar;
        ArmorEKF armor;
        armor.base_frame = true;
        Gimbal gimbal(gimbal_config);
        Output output;
        output.open(root, camera, predictor_type, preview);
        if (camera_source == CAMERA)
            serial.send_mode(HOST_AUTO_AIM);
        do
        {
            const auto frame_start = std::chrono::steady_clock::now();
            auto detections = detectArmors(camera.image_, target_color);
            const auto camera_poses = pnp.solve_frame(std::move(detections), camera.timestamp_ms);
            if (camera_source == VIDEO)
                pose_base.push({camera.timestamp_ms, 0, 0, 0, HOST_IDLE, 0});
            if (camera_source == CAMERA)
                pose_base.synchronize(serial, camera.timestamp_ms);
            const auto base_poses = pose_base.convert(camera_poses, camera.timestamp_ms);
            VideoPrediction prediction;
            if (predictor_type == PREDICTOR_SINGLE_PLATE)
                prediction = single_plate.update_frame(camera.frame_id_, camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            else if (predictor_type == PREDICTOR_POLAR)
                prediction = polar.update_frame(camera.frame_id_, camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            else
                prediction = armor.update_frame(camera.frame_id_, camera.timestamp_ms, base_poses.observations, prediction_horizon_ms);
            const auto command = gimbal.solve(prediction, pose_base, camera.timestamp_ms,
                camera_source == CAMERA ? monotonic_time_ms() : camera.timestamp_ms);
            if (camera_source == CAMERA)
                serial.send_target(command.yaw_rad, command.pitch_rad, command.valid);
            if (!output.write(camera, pnp, camera_poses, base_poses, pose_base, prediction, command, frame_start))
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
