#include "predictor.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

double single_plate_wrap_to_pi(double angle)
{
    double result = std::fmod(angle + single_plate_pi, 2 * single_plate_pi);
    return (result < 0 ? result + 2 * single_plate_pi : result) - single_plate_pi;
}

bool single_plate_valid_observation(const Eigen::Vector3d &z)
{
    return z(2) > single_plate_geometry_epsilon && std::abs(z(1)) < single_plate_pi / 2 &&
           z(2) * std::cos(z(1)) > single_plate_geometry_epsilon;
}

Eigen::Vector3d single_plate_h(const Eigen::Matrix<double, 6, 1> &s)
{
    const double horizontal = std::hypot(s(0), s(4));
    return {std::atan2(s(0), s(4)), std::atan2(s(2), horizontal), std::hypot(horizontal, s(2))};
}

Eigen::Matrix<double, 3, 6> single_plate_jacobian(const Eigen::Matrix<double, 6, 1> &s)
{
    const double x = s(0), y = s(2), z = s(4), horizontal = std::hypot(x, z),
                 distance = std::hypot(horizontal, y), horizontal2 = horizontal * horizontal,
                 distance2 = distance * distance;
    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
    H(0, 0) = z / horizontal2;
    H(0, 4) = -x / horizontal2;
    H(1, 0) = -x * y / (horizontal * distance2);
    H(1, 2) = horizontal / distance2;
    H(1, 4) = -y * z / (horizontal * distance2);
    H(2, 0) = x / distance;
    H(2, 2) = y / distance;
    H(2, 4) = z / distance;
    return H;
}

void SinglePlateEKF::initialize(const Eigen::Vector3d &z)
{
    const double horizontal = z(2) * std::cos(z(1));
    state_ << horizontal * std::sin(z(0)), 0, z(2) * std::sin(z(1)), 0,
        horizontal * std::cos(z(0)), 0;
    covariance_ = Eigen::Matrix<double, 6, 6>::Identity() * 10;
    initialized_ = true;
}

void SinglePlateEKF::predict(double dt)
{
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity(), Q = Eigen::Matrix<double, 6, 6>::Zero();
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    for (int i = 0; i < 6; i += 2)
    {
        F(i, i + 1) = dt;
        Q(i, i) = single_plate_process_noise * dt4 / 4;
        Q(i, i + 1) = Q(i + 1, i) = single_plate_process_noise * dt3 / 2;
        Q(i + 1, i + 1) = single_plate_process_noise * dt2;
    }
    const Eigen::Matrix<double, 6, 1> predicted = F * state_;
    const Eigen::Matrix<double, 6, 6> covariance = F * covariance_ * F.transpose() + Q;
    state_ = predicted;
    covariance_ = covariance;
}

bool SinglePlateEKF::update(const Eigen::Vector3d &z)
{
    if (std::hypot(state_(0), state_(4)) <= single_plate_geometry_epsilon)
        return false;
    const Eigen::Matrix<double, 3, 6> H = single_plate_jacobian(state_);
    Eigen::Vector3d residual = z - single_plate_h(state_);
    residual(0) = single_plate_wrap_to_pi(residual(0));
    residual(1) = single_plate_wrap_to_pi(residual(1));
    const Eigen::Matrix3d R = Eigen::Vector3d(.0016, .0016, .16).asDiagonal(),
                          S = H * covariance_ * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix3d> solver(S);
    const Eigen::Matrix<double, 6, 3> K = solver.solve(H * covariance_).transpose();
    const Eigen::Matrix<double, 6, 1> corrected = state_ + K * residual;
    const Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Identity() - K * H;
    Eigen::Matrix<double, 6, 6> covariance = A * covariance_ * A.transpose() + K * R * K.transpose();
    covariance = ((covariance + covariance.transpose()) * .5).eval();
    state_ = corrected;
    covariance_ = covariance;
    return true;
}


VideoPrediction SinglePlateEKF::update_frame(std::int64_t frame, double timestamp_ms,
                                            const std::vector<VideoObservation> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms_) / 1000;
    last_timestamp_ms_ = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    VideoPrediction result;
    result.horizon_ms = horizon_ms;
    const VideoObservation *selected = nullptr;
    for (const auto &observation : observations)
        if (single_plate_valid_observation(observation.measurement.head<3>()) &&
            (!selected || observation.measurement(2) < selected->measurement(2)))
            selected = &observation;

    if (!initialized_)
    {
        if (selected)
            initialize(selected->measurement.head<3>());
        result.status = selected ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        const auto predicted = single_plate_h(state_);
        auto &errors = result.errors;
        if (selected)
            errors << state_(0) - selected->position(0), state_(4) - selected->position(2),
                single_plate_wrap_to_pi(predicted(0) - selected->measurement(0)), predicted(2) - selected->measurement(2);
        result.values = {double(frame), state_(0), selected ? selected->position(0) : missing, errors(0),
                         state_(4), selected ? selected->position(2) : missing, errors(1), predicted(0),
                         selected ? selected->measurement(0) : missing, errors(2), predicted(2),
                         selected ? selected->measurement(2) : missing, errors(3)};
        result.value_count = 13;
        result.status = selected && update(selected->measurement.head<3>()) ? "updated" : "prediction_only";
    }
    result.initialized = initialized_;
    if (initialized_)
    {
        result.position_variance = std::max({covariance_(0, 0), covariance_(2, 2), covariance_(4, 4)});
        result.current.plates.emplace_back(state_(0), state_(2), state_(4), 0);
        const double horizon_s = horizon_ms / 1000;
        result.future.plates.emplace_back(state_(0) + horizon_s * state_(1), state_(2) + horizon_s * state_(3),
                                          state_(4) + horizon_s * state_(5), 0);
    }
    return result;
}
