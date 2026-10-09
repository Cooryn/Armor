#pragma once
#include <Eigen/Dense>
#include <vector>
#include "serial.hpp"
#include "solver.hpp"


struct CameraCalibration
{
    Eigen::Vector3d yaw_to_pitch = Eigen::Vector3d::Zero();
    Eigen::Vector3d camera_in_pitch = Eigen::Vector3d::Zero();
    Eigen::Quaterniond camera_to_pitch = Eigen::Quaterniond::Identity();
    Eigen::Vector3d angle_zero_rad = Eigen::Vector3d::Zero();
    Eigen::Vector3d angle_direction = Eigen::Vector3d::Ones();
};
struct CameraSample
{
    double timestamp_ms, yaw_deg, pitch_deg, roll_deg;
};
struct BaseArmorPose
{
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d rvec = Eigen::Vector3d::Zero();
    Eigen::Vector4d measurement = Eigen::Vector4d::Zero();
};
class PoseBase
{
public:
    PoseBase(const CameraCalibration &calibration = CameraCalibration{},
             const std::vector<CameraSample> &samples = {}, double max_gap_ms = 100)
    {
        calibration_ = calibration;
        samples_ = samples;
        max_gap_ms_ = max_gap_ms;
    }
    void receive(Serial &serial);
    void synchronize(Serial &serial, double timestamp_ms);
    bool covered(double timestamp_ms) const;
    SolvedFrame convert(const SolvedFrame &camera_poses, double timestamp_ms) const;
    Eigen::Isometry3d at_angles(const Eigen::Vector3d &raw_angles) const;
    void push(const GimbalState &gimbal_state);
    Eigen::Isometry3d at(double timestamp_ms) const;
    BaseArmorPose to_base(const Eigen::Vector3d &camera_position, const Eigen::Vector3d &camera_rvec,
                          double image_timestamp_ms) const;
    Eigen::Isometry3d reset_origin(double timestamp_ms);

    CameraCalibration calibration_;
    std::vector<CameraSample> samples_;
    double max_gap_ms_ = 100;
    Eigen::Vector3d origin_in_mechanical_ = Eigen::Vector3d::Zero();
    double reference_time_ms_ = 0;
    GimbalState latest_state_;

private:
    Eigen::Isometry3d mechanical_at(double timestamp_ms) const;
    Eigen::Isometry3d transform(const CameraSample &sample) const;
};
