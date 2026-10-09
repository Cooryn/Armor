#include "pose_base.hpp"
#include "serial.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>
#include <stdexcept>

Eigen::Isometry3d PoseBase::at(double timestamp_ms) const
{
    Eigen::Isometry3d pose = mechanical_at(timestamp_ms);
    pose.translation() -= origin_in_mechanical_;
    return pose;
}

Eigen::Isometry3d PoseBase::reset_origin(double timestamp_ms)
{
    const Eigen::Vector3d new_origin = mechanical_at(timestamp_ms).translation();
    Eigen::Isometry3d new_from_old = Eigen::Isometry3d::Identity();
    new_from_old.translation() = origin_in_mechanical_ - new_origin;
    origin_in_mechanical_ = new_origin;
    reference_time_ms_ = timestamp_ms;
    return new_from_old;
}

Eigen::Isometry3d PoseBase::mechanical_at(double timestamp_ms) const
{
    auto right = std::lower_bound(samples_.begin(), samples_.end(), timestamp_ms,
                                  [](const CameraSample &sample, double t)
                                  { return sample.timestamp_ms < t; });
    if (right->timestamp_ms == timestamp_ms)
        return transform(*right);
    const auto &left = *std::prev(right);
    const double f = (timestamp_ms - left.timestamp_ms) / (right->timestamp_ms - left.timestamp_ms);
    auto angle = [f](double a, double b)
    { return a + f * std::remainder(b - a, 360.0); };
    return transform({timestamp_ms, angle(left.yaw_deg, right->yaw_deg),
                      angle(left.pitch_deg, right->pitch_deg), angle(left.roll_deg, right->roll_deg)});
}

Eigen::Isometry3d PoseBase::transform(const CameraSample &sample) const
{
    constexpr double radians = 3.14159265358979323846 / 180;
    const Eigen::Matrix3d yaw = Eigen::AngleAxisd(sample.yaw_deg * radians, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d pitch_roll = (Eigen::AngleAxisd(sample.pitch_deg * radians, Eigen::Vector3d::UnitX()) *
                                        Eigen::AngleAxisd(sample.roll_deg * radians, Eigen::Vector3d::UnitZ()))
                                           .toRotationMatrix();
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = yaw * pitch_roll * calibration_.camera_to_pitch.toRotationMatrix();
    pose.translation() = yaw * (calibration_.yaw_to_pitch + pitch_roll * calibration_.camera_in_pitch);
    return pose;
}

void PoseBase::push(const GimbalState &gimbal_state)
{
    const Eigen::Vector3d raw_angles(gimbal_state.yaw_rad, gimbal_state.pitch_rad, gimbal_state.roll_rad);
    const Eigen::Vector3d angles = calibration_.angle_direction.cwiseProduct(raw_angles - calibration_.angle_zero_rad) *
                                   (180 / 3.14159265358979323846);
    const CameraSample sample{gimbal_state.receive_timestamp_ms, angles.x(), angles.y(), angles.z()};
    if (samples_.empty())
    {
        origin_in_mechanical_ = transform(sample).translation();
        reference_time_ms_ = sample.timestamp_ms;
    }
    latest_state_ = gimbal_state;
    samples_.push_back(sample);
    if (samples_.size() > 256)
        samples_.erase(samples_.begin(), samples_.end() - 256);
}

BaseArmorPose PoseBase::to_base(const Eigen::Vector3d &camera_position,
                                    const Eigen::Vector3d &camera_rvec, double image_timestamp_ms) const
{
    const auto base_from_camera = at(image_timestamp_ms);
    cv::Mat rotation;
    cv::Rodrigues(cv::Vec3d(camera_rvec.x(), camera_rvec.y(), camera_rvec.z()), rotation);
    const Eigen::Matrix3d camera_from_armor = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation.ptr<double>());
    const Eigen::Matrix3d base_from_armor = base_from_camera.linear() * camera_from_armor;
    BaseArmorPose result;
    result.position = base_from_camera * camera_position;
    const Eigen::AngleAxisd angle_axis(base_from_armor);
    result.rvec = angle_axis.axis() * angle_axis.angle();
    const Eigen::Vector3d normal = base_from_armor.col(2);
    const double distance = result.position.norm();

    result.measurement = {std::atan2(result.position.x(), result.position.z()),
                          std::atan2(result.position.y(), std::hypot(result.position.x(), result.position.z())),
                          distance, std::atan2(-normal.x(), normal.z())};
    return result;
}

bool PoseBase::covered(double timestamp_ms) const
{
    auto right = std::lower_bound(samples_.begin(), samples_.end(), timestamp_ms,
        [](const CameraSample &sample, double t) { return sample.timestamp_ms < t; });
    if (right != samples_.end() && right->timestamp_ms == timestamp_ms)
        return true;
    return right != samples_.begin() && right != samples_.end() &&
           right->timestamp_ms - std::prev(right)->timestamp_ms <= max_gap_ms_;
}

void PoseBase::receive(Serial &serial)
{
    GimbalState gimbal_state;
    const double deadline = monotonic_time_ms() + 100;
    while (!serial.receive(gimbal_state))
    {
        if (monotonic_time_ms() >= deadline)
            throw std::runtime_error("Timed out waiting for MCU STATE");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    push(gimbal_state);
}

void PoseBase::synchronize(Serial &serial, double timestamp_ms)
{
    while (!covered(timestamp_ms))
    {
        if (!samples_.empty() && samples_.back().timestamp_ms >= timestamp_ms)
            throw std::runtime_error("MCU STATE does not cover the image time");
        receive(serial);
    }
}

Eigen::Isometry3d PoseBase::at_angles(const Eigen::Vector3d &raw_angles) const
{
    const Eigen::Vector3d angles = calibration_.angle_direction.cwiseProduct(raw_angles - calibration_.angle_zero_rad) *
                                 (180 / 3.14159265358979323846);
    Eigen::Isometry3d pose = transform({0, angles.x(), angles.y(), angles.z()});
    pose.translation() -= origin_in_mechanical_;
    return pose;
}

SolvedFrame PoseBase::convert(const SolvedFrame &camera_poses, double timestamp_ms) const
{
    SolvedFrame result;
    for (std::size_t i = 0; i < camera_poses.armors.size(); ++i)
    {
        auto armor = camera_poses.armors[i];
        const auto pose = to_base(camera_poses.observations.at(i).position,
            {armor.rvec.at<double>(0), armor.rvec.at<double>(1), armor.rvec.at<double>(2)}, timestamp_ms);
        armor.tvec = (cv::Mat_<double>(3, 1) << pose.position.x(), pose.position.y(), pose.position.z());
        armor.rvec = (cv::Mat_<double>(3, 1) << pose.rvec.x(), pose.rvec.y(), pose.rvec.z());
        armor.yaw = pose.measurement(3) * 180 / CV_PI;
        result.armors.push_back(std::move(armor));
        result.observations.push_back({pose.position, pose.measurement});
    }
    return result;
}
