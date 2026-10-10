#include "gimbal.hpp"
#include "output.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

ControlTarget gimbal::solve(const PredictionResult &prediction, const ::pose &pose_base,
                                 double image_timestamp_ms, double now_ms, const GimbalConfig &gimbal_config)
{
    const auto &gimbal_state = pose_base.latest_state_;
    ControlTarget control;
    control.yaw_rad = gimbal_state.yaw_rad;
    control.pitch_rad = gimbal_state.pitch_rad;
    if (now_ms - gimbal_state.receive_timestamp_ms > gimbal_config.max_age_ms || now_ms - image_timestamp_ms > gimbal_config.max_age_ms)
    {
        control.status = "stale_pose";
        return control;
    }
    if (prediction.status != "updated" && prediction.status != "initialized")
        return control;
    const auto camera_from_base = pose_base.at(image_timestamp_ms).inverse();
    const Eigen::Vector4d *selected_plate = nullptr;
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < prediction.future.plates.size(); ++i)
    {
        const auto &plate = prediction.future.plates[i];
        const Eigen::Vector3d camera_position = camera_from_base * plate.head<3>();
        if (camera_position.z() > .1 && camera_position.norm() < nearest)
        {
            selected_plate = &plate;
            nearest = camera_position.norm();
            control.armor_id = static_cast<int>(i);
        }
    }
    if (!selected_plate)
        return control;
    Eigen::Vector3d angles(gimbal_state.yaw_rad, gimbal_state.pitch_rad, gimbal_state.roll_rad);
    const auto residual = [&](const Eigen::Vector3d &candidate)
    {
        const Eigen::Vector3d camera_position = pose_base.at_angles(candidate).inverse() * selected_plate->head<3>();
        return Eigen::Vector2d(std::atan2(camera_position.x(), camera_position.z()), std::atan2(camera_position.y(), std::hypot(camera_position.x(), camera_position.z())));
    };
    bool converged = false;
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        const Eigen::Vector2d error = residual(angles);
        if (error.norm() < 1e-7)
        {
            converged = true;
            break;
        }
        Eigen::Matrix2d jacobian;
        constexpr double step = 1e-5;
        for (int axis = 0; axis < 2; ++axis)
        {
            auto plus = angles, minus = angles;
            plus(axis) += step;
            minus(axis) -= step;
            Eigen::Vector2d difference = residual(plus) - residual(minus);
            difference(0) = std::remainder(difference(0), 2 * CV_PI);
            jacobian.col(axis) = difference / (2 * step);
        }
        Eigen::FullPivLU<Eigen::Matrix2d> decomposition(jacobian);
        const Eigen::Vector2d correction = decomposition.solve(error);
        angles.head<2>() -= correction * std::min(1., .25 / correction.norm());
    }
    if (!converged)
        throw std::runtime_error("Optical alignment did not converge");
    angles.x() = gimbal_state.yaw_rad + std::remainder(angles.x() - gimbal_state.yaw_rad, 2 * CV_PI);
    if (angles.x() < gimbal_config.yaw_min_rad || angles.x() > gimbal_config.yaw_max_rad ||
        angles.y() < gimbal_config.pitch_min_rad || angles.y() > gimbal_config.pitch_max_rad)
    {
        control.status = "out_of_limits";
        return control;
    }
    control.yaw_rad = static_cast<float>(angles.x());
    control.pitch_rad = static_cast<float>(angles.y());
    control.status = "tracking";
    return control;
}
