#include "pose_base.hpp"
#include <cmath>
#include <stdexcept>

void pose::update(const GimbalState &gimbal_state)
{
    const Eigen::Vector3d raw_angles(gimbal_state.yaw_rad, gimbal_state.pitch_rad, gimbal_state.roll_rad);
    const Eigen::Vector3d angles = calibration_.angle_direction.cwiseProduct(raw_angles - calibration_.angle_zero_rad);
    if (!initialized_)
    {
        const Eigen::Vector3d forward = transform(angles).linear().col(2);
        if (std::hypot(forward.x(), forward.z()) < 1e-6)
            throw std::invalid_argument("Initial camera direction has no horizontal projection");
        reference_yaw_rad_ = std::atan2(forward.x(), forward.z());
        reference_time_ms_ = gimbal_state.receive_timestamp_ms;
    }
    base_from_camera_ = transform(angles);
    latest_state_ = gimbal_state;
    initialized_ = true;
}

Eigen::Isometry3d pose::transform(const Eigen::Vector3d &angles) const
{
    const Eigen::Matrix3d yaw = Eigen::AngleAxisd(angles.x() - reference_yaw_rad_,
                                                  Eigen::Vector3d::UnitY())
                                    .toRotationMatrix();
    const Eigen::Matrix3d pitch_roll = (Eigen::AngleAxisd(angles.y(), Eigen::Vector3d::UnitX()) *
                                        Eigen::AngleAxisd(angles.z(), Eigen::Vector3d::UnitZ()))
                                           .toRotationMatrix();
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() = yaw * pitch_roll * calibration_.camera_to_pitch.toRotationMatrix();
    result.translation() = yaw * (calibration_.yaw_to_pitch + pitch_roll * calibration_.camera_in_pitch);
    return result;
}

Eigen::Isometry3d pose::at_angles(const Eigen::Vector3d &raw_angles) const
{
    if (!initialized_)
        throw std::logic_error("Base heading requires an initial camera attitude");
    const Eigen::Vector3d angles = calibration_.angle_direction.cwiseProduct(raw_angles - calibration_.angle_zero_rad);
    return transform(angles);
}

SolvedFrame pose::convert(const SolvedFrame &camera_poses) const
{
    if (!initialized_)
        throw std::logic_error("Camera attitude has not been initialized");
    SolvedFrame result;
    for (const auto &camera_armor : camera_poses.armors)
    {
        auto armor = camera_armor;
        const Eigen::Vector3d position = base_from_camera_ * Eigen::Vector3d(
                                                                 armor.tvec.at<double>(0), armor.tvec.at<double>(1), armor.tvec.at<double>(2));
        cv::Mat rotation;
        cv::Rodrigues(armor.rvec, rotation);
        const Eigen::Matrix3d camera_rotation =
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation.ptr<double>());
        const Eigen::Matrix3d base_rotation = base_from_camera_.linear() * camera_rotation;
        const Eigen::AngleAxisd angle_axis(base_rotation);
        const Eigen::Vector3d rvec = angle_axis.axis() * angle_axis.angle();
        const Eigen::Vector3d normal = base_rotation.col(2);
        const Eigen::Vector4d measurement(
            std::atan2(position.x(), position.z()),
            std::atan2(position.y(), std::hypot(position.x(), position.z())),
            position.norm(), std::atan2(-normal.x(), normal.z()));
        armor.tvec = (cv::Mat_<double>(3, 1) << position.x(), position.y(), position.z());
        armor.rvec = (cv::Mat_<double>(3, 1) << rvec.x(), rvec.y(), rvec.z());
        armor.yaw = measurement(3) * 180 / CV_PI;
        result.armors.push_back(std::move(armor));
        result.observations.push_back(measurement);
    }
    return result;
}
