#include "predictor_polar.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

double predictor_polar::wrap_to_pi(double angle)
{
    double wrapped_angle = std::fmod(angle + predictor_polar::pi, 2 * predictor_polar::pi);
    if (wrapped_angle < 0)
        wrapped_angle += 2 * predictor_polar::pi;
    return wrapped_angle - predictor_polar::pi;
}

Eigen::Vector4d predictor_polar::angular_residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted)
{
    Eigen::Vector4d residual = observed - predicted;
    for (int i : {0, 1, 3})
        residual(i) = predictor_polar::wrap_to_pi(residual(i));
    return residual;
}

int predictor_polar::round_even(double x)
{
    double lo = std::floor(x), f = x - lo;
    return static_cast<int>(f < .5                            ? lo
                            : f > .5                          ? lo + 1
                            : std::fmod(std::abs(lo), 2) == 0 ? lo
                                                              : lo + 1);
}

void predictor_polar::predict(double dt)
{
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    for (int i : {0, 2, 4, 6})
        F(i, i + 1) = dt;
    X = F * X;
    X(6) = predictor_polar::wrap_to_pi(X(6));
    P = F * P * F.transpose() + Q;
}

Eigen::Vector4d predictor_polar::h(const Eigen::Matrix<double, 9, 1> &state, int armor_id)
{
    double yaw = state(6) + armor_id * predictor_polar::pi / 2, x = state(0) + state(8) * std::sin(yaw),
           z = state(4) - state(8) * std::cos(yaw), y = state(2);
    return {predictor_polar::wrap_to_pi(std::atan2(x, z)),
            predictor_polar::wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), predictor_polar::wrap_to_pi(yaw)};
}

Eigen::Matrix<double, 4, 9> predictor_polar::get_jacobian(const Eigen::Matrix<double, 9, 1> &state, int armor_id)
{
    Eigen::Matrix<double, 4, 9> H = Eigen::Matrix<double, 4, 9>::Zero();
    const Eigen::Vector4d predicted_observation = predictor_polar::h(state, armor_id);
    for (int i = 0; i < 9; ++i)
    {
        Eigen::Matrix<double, 9, 1> perturbed_state = state;
        perturbed_state(i) += 1e-5;
        const Eigen::Vector4d residual = predictor_polar::angular_residual(predictor_polar::h(perturbed_state, armor_id), predicted_observation);
        H.col(i) = residual / 1e-5;
    }
    return H;
}

void predictor_polar::update(const Eigen::Vector4d &z)
{
    int armor_id = predictor_polar::round_even(predictor_polar::wrap_to_pi(z(3) - X(6)) / (predictor_polar::pi / 2));
    const Eigen::Matrix<double, 4, 9> H = predictor_polar::get_jacobian(X, armor_id);
    const Eigen::Vector4d residual = predictor_polar::angular_residual(z, predictor_polar::h(X, armor_id));
    const Eigen::Matrix4d S = H * P * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix4d> decomposition(S);
    const Eigen::Matrix<double, 9, 4> PHt = P * H.transpose();
    const Eigen::Matrix<double, 9, 4> K = decomposition.solve(PHt.transpose()).transpose();
    X = X + K * residual;
    X(6) = predictor_polar::wrap_to_pi(X(6));
    P = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * P;
}

PredictionResult predictor_polar::update_frame(double timestamp_ms,
                                      const std::vector<Eigen::Vector4d> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms) / 1000;
    last_timestamp_ms = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    PredictionResult prediction;
    const Eigen::Vector4d *selected_observation = nullptr;
    for (const auto &observation : observations)
    {
        const auto &z = observation;
        if (z(2) > 1e-6 && std::abs(z(1)) < predictor_polar::pi / 2 && z(2) * std::cos(z(1)) > 1e-6 &&
            (!selected_observation || z(2) < (*selected_observation)(2)))
            selected_observation = &observation;
    }
    if (!is_initialized)
    {
        if (selected_observation)
        {
            const auto &z = (*selected_observation);
            const Eigen::Vector3d position(z(2) * std::cos(z(1)) * std::sin(z(0)), z(2) * std::sin(z(1)),
                                           z(2) * std::cos(z(1)) * std::cos(z(0)));
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
            const int armor_id = predictor_polar::round_even(predictor_polar::wrap_to_pi((*selected_observation)(3) - X(6)) / (predictor_polar::pi / 2));
            prediction.errors = predictor_polar::angular_residual(predictor_polar::h(X, armor_id), (*selected_observation));
            update((*selected_observation));
        }
        prediction.values = {X(1), X(3), X(5), X(6), X(7), X(8),
                             selected_observation ? (*selected_observation)(3) : missing};
        prediction.status = selected_observation ? "updated" : "prediction_only";
    }
    if (is_initialized)
    {
        auto future = X;
        for (int i : {0, 2, 4, 6})
            future(i) += horizon_ms / 1000 * future(i + 1);
        future(6) = predictor_polar::wrap_to_pi(future(6));
        for (bool forecast : {false, true})
        {
            const auto &state = forecast ? future : X;
            auto &geometry = forecast ? prediction.future : prediction.current;
            geometry.center = Eigen::Vector3d(state(0), state(2), state(4));
            for (int i = 0; i < 4; ++i)
            {
                const double yaw = predictor_polar::wrap_to_pi(state(6) + i * predictor_polar::pi / 2);
                geometry.plates.emplace_back(state(0) + state(8) * std::sin(yaw), state(2), state(4) - state(8) * std::cos(yaw), yaw);
            }
        }
    }
    return prediction;
}
