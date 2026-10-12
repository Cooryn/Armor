#pragma once
#include <Eigen/Dense>
#include "serial.hpp"
#include "solver.hpp"

struct CameraCalibration
{
    // 安装位移单位为米；未标定时的零偏移、单位旋转只是临时近似。
    Eigen::Vector3d yaw_to_pitch = Eigen::Vector3d::Zero();
    Eigen::Vector3d camera_in_pitch = Eigen::Vector3d::Zero();
    Eigen::Quaterniond camera_to_pitch = Eigen::Quaterniond::Identity();
    Eigen::Vector3d angle_zero_rad = Eigen::Vector3d::Zero();
    Eigen::Vector3d angle_direction = Eigen::Vector3d::Ones();
};

class pose
{
public:
    pose(const CameraCalibration &calibration = CameraCalibration{}) : calibration_(calibration) {}
    void update(const GimbalState &gimbal_state);
    SolvedFrame convert(const SolvedFrame &camera_poses) const;
    Eigen::Isometry3d at_angles(const Eigen::Vector3d &raw_angles) const;

    CameraCalibration calibration_;
    Eigen::Isometry3d base_from_camera_ = Eigen::Isometry3d::Identity();
    GimbalState latest_state_;
    // 基座原点固定在 yaw 转轴中心，Y 向下，Z 沿初始视线的水平投影。
    double reference_time_ms_ = 0;
    double reference_yaw_rad_ = 0;
    bool initialized_ = false;

private:
    Eigen::Isometry3d transform(const Eigen::Vector3d &angles) const;
};
