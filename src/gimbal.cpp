#include "gimbal.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

ControlTarget Gimbal::solve(const VideoPrediction &prediction, const PoseBase &pose_base,
                                 double image_timestamp_ms, double now_ms) const
{
    const auto &state = pose_base.latest_state_;
    ControlTarget result;
    result.yaw_rad = state.yaw_rad;
    result.pitch_rad = state.pitch_rad;
    if (now_ms - state.receive_timestamp_ms > config_.max_age_ms || now_ms - image_timestamp_ms > config_.max_age_ms)
    {
        result.status = "stale_pose";
        return result;
    }
    if (!prediction.initialized || (prediction.status != "updated" && prediction.status != "initialized"))
        return result; // A missed or rejected observation sends valid=false immediately.
    if (prediction.position_variance > config_.max_position_variance)
    {
        result.status = "uncertain";
        return result;
    }
    const auto camera_from_base = pose_base.at(image_timestamp_ms).inverse();
    const Eigen::Vector4d *selected = nullptr;
    double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < prediction.future.plates.size(); ++i)
    {
        const auto &plate = prediction.future.plates[i];
        const Eigen::Vector3d camera_position = camera_from_base * plate.head<3>();
        if (camera_position.z() > .1 && camera_position.norm() < nearest)
        {
            selected = &plate;
            nearest = camera_position.norm();
            result.armor_id = static_cast<int>(i);
        }
    }
    if (!selected)
        return result;
    Eigen::Vector3d angles(state.yaw_rad, state.pitch_rad, state.roll_rad);
    // Solve optical alignment with the same offsets, mount, zero and signs as PoseBase.
    // Roll is measured but has no control channel. A 2x2 Newton solve finds yaw/pitch.
    const auto residual = [&](const Eigen::Vector3d &candidate)
    {
        const Eigen::Vector3d p = pose_base.at_angles(candidate).inverse() * selected->head<3>();
        return Eigen::Vector2d(std::atan2(p.x(), p.z()), std::atan2(p.y(), std::hypot(p.x(), p.z())));
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
    angles.x() = state.yaw_rad + std::remainder(angles.x() - state.yaw_rad, 2 * CV_PI);
    if (angles.x() < config_.yaw_min_rad || angles.x() > config_.yaw_max_rad ||
        angles.y() < config_.pitch_min_rad || angles.y() > config_.pitch_max_rad)
    {
        result.status = "out_of_limits";
        return result;
    }
    result.yaw_rad = static_cast<float>(angles.x());
    result.pitch_rad = static_cast<float>(angles.y());
    result.valid = true;
    result.status = "tracking";
    return result;
}
