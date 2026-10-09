#include "predictor_polar.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

double polar_wrap_to_pi(double angle)
{
    double wrapped_angle = std::fmod(angle + polar_pi, 2 * polar_pi);
    if (wrapped_angle < 0)
        wrapped_angle += 2 * polar_pi;
    return wrapped_angle - polar_pi;
}

Eigen::Vector4d polar_angular_residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted)
{
    Eigen::Vector4d residual = observed - predicted;
    for (int i : {0, 1, 3})
        residual(i) = polar_wrap_to_pi(residual(i));
    return residual;
}

int polar_round_even(double x)
{
    double lo = std::floor(x), f = x - lo;
    return static_cast<int>(f < .5                            ? lo
                            : f > .5                          ? lo + 1
                            : std::fmod(std::abs(lo), 2) == 0 ? lo
                                                              : lo + 1);
}

void PolarEKF::predict(double dt)
{
    for (int i : {0, 2, 4, 6})
        F(i, i + 1) = dt;
    X = F * X;
    X(6) = polar_wrap_to_pi(X(6));
    P = F * P * F.transpose() + Q;
}

Eigen::Vector4d polar_h(const Eigen::Matrix<double, 9, 1> &state, int armor_id)
{
    double yaw = state(6) + armor_id * polar_pi / 2, x = state(0) + state(8) * std::sin(yaw),
           z = state(4) - state(8) * std::cos(yaw), y = state(2);
    return {polar_wrap_to_pi(std::atan2(x, z)),
            polar_wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), polar_wrap_to_pi(yaw)};
}

Eigen::Matrix<double, 4, 9> polar_get_jacobian(const Eigen::Matrix<double, 9, 1> &state, int armor_id)
{
    Eigen::Matrix<double, 4, 9> H = Eigen::Matrix<double, 4, 9>::Zero();
    const Eigen::Vector4d predicted_observation = polar_h(state, armor_id);
    for (int i = 0; i < 9; ++i)
    {
        Eigen::Matrix<double, 9, 1> perturbed_state = state;
        perturbed_state(i) += 1e-5;
        const Eigen::Vector4d residual = polar_angular_residual(polar_h(perturbed_state, armor_id), predicted_observation);
        H.col(i) = residual / 1e-5;
    }
    return H;
}

void PolarEKF::update(const Eigen::Vector4d &z)
{
    int armor_id = polar_round_even(polar_wrap_to_pi(z(3) - X(6)) / (polar_pi / 2));
    const Eigen::Matrix<double, 4, 9> H = polar_get_jacobian(X, armor_id);
    const Eigen::Vector4d residual = polar_angular_residual(z, polar_h(X, armor_id));
    const Eigen::Matrix4d S = H * P * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix4d> decomposition(S);
    const Eigen::Matrix<double, 9, 4> PHt = P * H.transpose();
    const Eigen::Matrix<double, 9, 4> K = decomposition.solve(PHt.transpose()).transpose();
    X = X + K * residual;
    X(6) = polar_wrap_to_pi(X(6));
    P = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * P;
}

VideoPrediction PolarEKF::update_frame(std::int64_t frame_id, double timestamp_ms,
                                      const std::vector<VideoObservation> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms) / 1000;
    last_timestamp_ms = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    VideoPrediction prediction;
    prediction.horizon_ms = horizon_ms;
    const VideoObservation *selected_observation = nullptr;
    for (const auto &observation : observations)
    {
        const auto &z = observation.measurement;
        if (z(2) > 1e-6 && std::abs(z(1)) < polar_pi / 2 && z(2) * std::cos(z(1)) > 1e-6 &&
            (!selected_observation || z(2) < selected_observation->measurement(2)))
            selected_observation = &observation;
    }
    if (!is_initialized)
    {
        if (selected_observation)
        {
            const auto &position = selected_observation->position;
            const auto &z = selected_observation->measurement;
            X << position(0) - .26 * std::sin(z(3)), 0, position(1), 0,
                 position(2) + .26 * std::cos(z(3)), 0, z(3), 0, .26;
            is_initialized = true;
        }
        prediction.status = selected_observation ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        if (selected_observation)
        {
            const int armor_id = polar_round_even(polar_wrap_to_pi(selected_observation->measurement(3) - X(6)) / (polar_pi / 2));
            prediction.errors = polar_angular_residual(polar_h(X, armor_id), selected_observation->measurement);
            update(selected_observation->measurement);
        }
        const auto &errors = prediction.errors;
        prediction.values = {double(frame_id), X(0), X(1), X(2), X(3), X(4), X(5), X(6), X(7), X(8),
                         errors(0), errors(1), errors(2), errors(3), selected_observation ? selected_observation->measurement(3) : missing};
        prediction.value_count = 15;
        prediction.status = selected_observation ? "updated" : "prediction_only";
    }
    prediction.initialized = is_initialized;
    if (is_initialized)
    {
        prediction.position_variance = std::max({P(0, 0), P(2, 2), P(4, 4)});
        auto future = X;
        for (int i : {0, 2, 4, 6})
            future(i) += horizon_ms / 1000 * future(i + 1);
        future(6) = polar_wrap_to_pi(future(6));
        for (bool forecast : {false, true})
        {
            const auto &state = forecast ? future : X;
            auto &geometry = forecast ? prediction.future : prediction.current;
            geometry.center = Eigen::Vector3d(state(0), state(2), state(4));
            for (int i = 0; i < 4; ++i)
            {
                const double yaw = polar_wrap_to_pi(state(6) + i * polar_pi / 2);
                geometry.plates.emplace_back(state(0) + state(8) * std::sin(yaw), state(2), state(4) - state(8) * std::cos(yaw), yaw);
            }
        }
    }
    return prediction;
}
