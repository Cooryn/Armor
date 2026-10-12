#include "predictor_single.hpp"

#include <Eigen/Dense>
#include <cmath>

const double predictor_single::pi = 3.14159265358979323846;

double predictor_single::wrap_to_pi(double angle)
{
    double wrapped_angle = std::fmod(angle + predictor_single::pi, 2 * predictor_single::pi);
    if (wrapped_angle < 0)
        wrapped_angle += 2 * predictor_single::pi;
    return wrapped_angle - predictor_single::pi;
}

Eigen::Vector4d predictor_single::angular_residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted)
{
    Eigen::Vector4d residual = observed - predicted;
    for (int i : {0, 1, 3})
        residual(i) = predictor_single::wrap_to_pi(residual(i));
    return residual;
}

void predictor_single::predict(double dt)
{
    Eigen::Matrix<double, 8, 8> F = Eigen::Matrix<double, 8, 8>::Identity(), Q = Eigen::Matrix<double, 8, 8>::Zero();
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    for (int i = 0; i < 8; i += 2)
    {
        const double q = i == 6 ? q_yaw : q_pos;
        F(i, i + 1) = dt;
        Q(i, i) = q * dt4 / 4;
        Q(i, i + 1) = Q(i + 1, i) = q * dt3 / 2;
        Q(i + 1, i + 1) = q * dt2;
    }
    X = F * X;
    X(6) = predictor_single::wrap_to_pi(X(6));
    P = F * P * F.transpose() + Q;
}

Eigen::Vector4d predictor_single::h(const Eigen::Matrix<double, 8, 1> &state)
{
    const double horizontal = std::hypot(state(0), state(4));
    return {std::atan2(state(0), state(4)), std::atan2(state(2), horizontal), std::hypot(horizontal, state(2)), predictor_single::wrap_to_pi(state(6))};
}

Eigen::Matrix<double, 4, 8> predictor_single::jacobian(const Eigen::Matrix<double, 8, 1> &state)
{
    const double x = state(0), y = state(2), z = state(4), horizontal = std::hypot(x, z),
                 distance = std::hypot(horizontal, y), horizontal2 = horizontal * horizontal,
                 distance2 = distance * distance;
    Eigen::Matrix<double, 4, 8> H = Eigen::Matrix<double, 4, 8>::Zero();
    H(0, 0) = z / horizontal2;
    H(0, 4) = -x / horizontal2;
    H(1, 0) = -x * y / (horizontal * distance2);
    H(1, 2) = horizontal / distance2;
    H(1, 4) = -y * z / (horizontal * distance2);
    H(2, 0) = x / distance;
    H(2, 2) = y / distance;
    H(2, 4) = z / distance;
    H(3, 6) = 1;
    return H;
}

void predictor_single::initialize(const Eigen::Vector4d &z)
{
    const double horizontal = z(2) * std::cos(z(1));
    X << horizontal * std::sin(z(0)), 0, z(2) * std::sin(z(1)), 0,
        horizontal * std::cos(z(0)), 0, predictor_single::wrap_to_pi(z(3)), 0;
    P = Eigen::Matrix<double, 8, 8>::Identity() * 10;
    is_initialized = true;
}

bool predictor_single::update(const Eigen::Vector4d &z)
{
    if (std::hypot(X(0), X(4)) <= geometry_epsilon)
        return false;
    const Eigen::Matrix<double, 4, 8> H = predictor_single::jacobian(X);
    const Eigen::Vector4d residual = predictor_single::angular_residual(z, predictor_single::h(X));
    const Eigen::Matrix4d S = H * P * H.transpose() + R;
    const auto decomposition = S.ldlt();
    const Eigen::Matrix<double, 8, 4> K = decomposition.solve(H * P).transpose();
    X = X + K * residual;
    X(6) = predictor_single::wrap_to_pi(X(6));
    const Eigen::Matrix<double, 8, 8> A = Eigen::Matrix<double, 8, 8>::Identity() - K * H;
    P = A * P * A.transpose() + K * R * K.transpose();
    P = ((P + P.transpose()) * .5).eval();
    return true;
}

PredictionResult predictor_single::update_frame(double timestamp_ms,
                                                const std::vector<Eigen::Vector4d> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms) / 1000;
    last_timestamp_ms = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    PredictionResult prediction;
    const Eigen::Vector4d *selected_observation = nullptr;
    for (const auto &observation : observations)
    {
        if (!selected_observation || observation(2) < (*selected_observation)(2))
            selected_observation = &observation;
    }

    Eigen::Vector3d observed_position = Eigen::Vector3d::Constant(missing);
    if (selected_observation)
    {
        const auto &z = *selected_observation;
        observed_position = {z(2) * std::cos(z(1)) * std::sin(z(0)), z(2) * std::sin(z(1)),
                             z(2) * std::cos(z(1)) * std::cos(z(0))};
    }
    if (!is_initialized)
    {
        if (selected_observation)
            initialize(*selected_observation);
        prediction.status = selected_observation ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        const auto predicted_observation = predictor_single::h(X);
        if (selected_observation)
        {
            const auto residual = predictor_single::angular_residual(predicted_observation, *selected_observation);
            prediction.errors << X(0) - observed_position(0), X(4) - observed_position(2), residual(0), residual(2);
        }
        prediction.values = {X(0), observed_position(0), X(4), observed_position(2), predicted_observation(0),
                             selected_observation ? (*selected_observation)(0) : missing, predicted_observation(2),
                             selected_observation ? (*selected_observation)(2) : missing};
        prediction.status = selected_observation && update(*selected_observation) ? "updated" : "prediction_only";
    }
    if (is_initialized)
    {
        prediction.current.plates.emplace_back(X(0), X(2), X(4), X(6));
        const double horizon_s = horizon_ms / 1000;
        prediction.future.plates.emplace_back(X(0) + horizon_s * X(1), X(2) + horizon_s * X(3),
                                              X(4) + horizon_s * X(5), predictor_single::wrap_to_pi(X(6) + horizon_s * X(7)));
    }
    return prediction;
}
