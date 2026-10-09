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

Eigen::Vector3d single_plate_h(const Eigen::Matrix<double, 6, 1> &state)
{
    const double horizontal = std::hypot(state(0), state(4));
    return {std::atan2(state(0), state(4)), std::atan2(state(2), horizontal), std::hypot(horizontal, state(2))};
}

Eigen::Matrix<double, 3, 6> single_plate_jacobian(const Eigen::Matrix<double, 6, 1> &state)
{
    const double x = state(0), y = state(2), z = state(4), horizontal = std::hypot(x, z),
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
    X << horizontal * std::sin(z(0)), 0, z(2) * std::sin(z(1)), 0,
        horizontal * std::cos(z(0)), 0;
    P = Eigen::Matrix<double, 6, 6>::Identity() * 10;
    is_initialized = true;
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
    const Eigen::Matrix<double, 6, 1> predicted = F * X;
    const Eigen::Matrix<double, 6, 6> covariance = F * P * F.transpose() + Q;
    X = predicted;
    P = covariance;
}

bool SinglePlateEKF::update(const Eigen::Vector3d &z)
{
    if (std::hypot(X(0), X(4)) <= single_plate_geometry_epsilon)
        return false;
    const Eigen::Matrix<double, 3, 6> H = single_plate_jacobian(X);
    Eigen::Vector3d residual = z - single_plate_h(X);
    residual(0) = single_plate_wrap_to_pi(residual(0));
    residual(1) = single_plate_wrap_to_pi(residual(1));
    const Eigen::Matrix3d R = Eigen::Vector3d(.0016, .0016, .16).asDiagonal(),
                          S = H * P * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix3d> decomposition(S);
    const Eigen::Matrix<double, 6, 3> K = decomposition.solve(H * P).transpose();
    const Eigen::Matrix<double, 6, 1> corrected = X + K * residual;
    const Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Identity() - K * H;
    Eigen::Matrix<double, 6, 6> covariance = A * P * A.transpose() + K * R * K.transpose();
    covariance = ((covariance + covariance.transpose()) * .5).eval();
    X = corrected;
    P = covariance;
    return true;
}


VideoPrediction SinglePlateEKF::update_frame(std::int64_t frame_id, double timestamp_ms,
                                            const std::vector<VideoObservation> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms) / 1000;
    last_timestamp_ms = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    VideoPrediction prediction;
    prediction.horizon_ms = horizon_ms;
    const VideoObservation *selected_observation = nullptr;
    for (const auto &observation : observations)
        if (single_plate_valid_observation(observation.measurement.head<3>()) &&
            (!selected_observation || observation.measurement(2) < selected_observation->measurement(2)))
            selected_observation = &observation;

    if (!is_initialized)
    {
        if (selected_observation)
            initialize(selected_observation->measurement.head<3>());
        prediction.status = selected_observation ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        const auto predicted = single_plate_h(X);
        auto &errors = prediction.errors;
        if (selected_observation)
            errors << X(0) - selected_observation->position(0), X(4) - selected_observation->position(2),
                single_plate_wrap_to_pi(predicted(0) - selected_observation->measurement(0)), predicted(2) - selected_observation->measurement(2);
        prediction.values = {double(frame_id), X(0), selected_observation ? selected_observation->position(0) : missing, errors(0),
                         X(4), selected_observation ? selected_observation->position(2) : missing, errors(1), predicted(0),
                         selected_observation ? selected_observation->measurement(0) : missing, errors(2), predicted(2),
                         selected_observation ? selected_observation->measurement(2) : missing, errors(3)};
        prediction.value_count = 13;
        prediction.status = selected_observation && update(selected_observation->measurement.head<3>()) ? "updated" : "prediction_only";
    }
    prediction.initialized = is_initialized;
    if (is_initialized)
    {
        prediction.position_variance = std::max({P(0, 0), P(2, 2), P(4, 4)});
        prediction.current.plates.emplace_back(X(0), X(2), X(4), 0);
        const double horizon_s = horizon_ms / 1000;
        prediction.future.plates.emplace_back(X(0) + horizon_s * X(1), X(2) + horizon_s * X(3),
                                          X(4) + horizon_s * X(5), 0);
    }
    return prediction;
}
