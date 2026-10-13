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

Eigen::Vector4d predictor_single::residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted)
{
    Eigen::Vector4d residual = observed - predicted;
    for (int i : {0, 1, 3})
        residual(i) = predictor_single::wrap_to_pi(residual(i));
    return residual;
}

void predictor_single::transition(double dt, Eigen::Matrix<double, 8, 8> &F, Eigen::Matrix<double, 8, 8> &Q) const
{
    F.setIdentity();
    Q.setZero();
    const double p00 = dt * dt * dt * dt / 4 * q_pos;
    const double p01 = dt * dt * dt / 2 * q_pos;
    const double p11 = dt * dt * q_pos;
    const double y00 = dt * dt * dt * dt / 4 * q_yaw;
    const double y01 = dt * dt * dt / 2 * q_yaw;
    const double y11 = dt * dt * q_yaw;
    for (int i : {0, 2, 4, 6})
    {
        F(i, i + 1) = dt;
        Q(i, i) = i == 6 ? y00 : p00;
        Q(i, i + 1) = Q(i + 1, i) = i == 6 ? y01 : p01;
        Q(i + 1, i + 1) = i == 6 ? y11 : p11;
    }
}

void predictor_single::predict(double dt)
{
    Eigen::Matrix<double, 8, 8> F, Q;
    transition(dt, F, Q);
    X = F * X;
    X(6) = predictor_single::wrap_to_pi(X(6));
    P = F * P * F.transpose() + Q;
}

Eigen::Vector4d predictor_single::plate_pose(const Eigen::Matrix<double, 8, 1> &X)
{
    double yaw = X(6);
    double x = X(0);
    double y = X(2);
    double z = X(4);
    return {x, y, z, predictor_single::wrap_to_pi(yaw)};
}

Eigen::Vector4d predictor_single::h(const Eigen::Matrix<double, 8, 1> &X)
{
    const auto pose = predictor_single::plate_pose(X);
    const double x = pose(0), y = pose(1), z = pose(2);
    const double horizontal = std::hypot(x, z);
    return {predictor_single::wrap_to_pi(std::atan2(x, z)), predictor_single::wrap_to_pi(std::atan2(y, horizontal)), std::hypot(horizontal, y), pose(3)};
}

Eigen::Matrix<double, 4, 8> predictor_single::jacobian(const Eigen::Matrix<double, 8, 1> &X)
{
    Eigen::Matrix<double, 4, 8> H = Eigen::Matrix<double, 4, 8>::Zero();
    const Eigen::Vector4d predicted_observation = predictor_single::h(X);
    for (int i : {0, 2, 4, 6})
    {
        Eigen::Matrix<double, 8, 1> perturbed_state = X;
        perturbed_state(i) += 1e-5;
        const Eigen::Vector4d residual = predictor_single::residual(predictor_single::h(perturbed_state), predicted_observation);
        H.col(i) = residual / 1e-5;
    }
    return H;
}

void predictor_single::initialize(const Eigen::Vector4d &Z)
{
    const double horizontal = Z(2) * std::cos(Z(1));
    const double x = horizontal * std::sin(Z(0)), y = Z(2) * std::sin(Z(1)),
                 z = horizontal * std::cos(Z(0));
    const double yaw = predictor_single::wrap_to_pi(Z(3));
    X << x, 0, y, 0, z, 0, yaw, 0;
    P = Eigen::Matrix<double, 8, 8>::Identity() * 10;
    is_initialized = true;
}

bool predictor_single::update(const Eigen::Vector4d &Z)
{
    if (std::hypot(X(0), X(4)) <= 1e-6)
        return false;
    const Eigen::Matrix<double, 4, 8> H = predictor_single::jacobian(X);
    const Eigen::Vector4d residual = predictor_single::residual(Z, predictor_single::h(X));
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

PredictionResult predictor_single::update_frame(double timestamp_ms, const std::vector<Eigen::Vector4d> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms) / 1000;
    last_timestamp_ms = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    PredictionResult prediction;
    const Eigen::Vector4d *selected_observation = nullptr;

    for (const auto &Z : observations)
    {
        if (!selected_observation || Z(2) < (*selected_observation)(2))
            selected_observation = &Z;
    }

    Eigen::Vector4d plate = Eigen::Vector4d::Constant(missing);
    Eigen::Vector4d predicted_observation = Eigen::Vector4d::Constant(missing);
    Eigen::Vector3d observed_position = Eigen::Vector3d::Constant(missing);

    if (!is_initialized)
    {
        if (selected_observation)
            initialize(*selected_observation);
        prediction.status = selected_observation ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        const Eigen::Matrix<double, 8, 1> prior_state = X;
        const bool updated = selected_observation && update(*selected_observation);
        plate = predictor_single::plate_pose(prior_state);
        predicted_observation = predictor_single::h(prior_state);
        if (selected_observation)
        {
            const auto &Z = *selected_observation;
            observed_position = {Z(2) * std::cos(Z(1)) * std::sin(Z(0)), Z(2) * std::sin(Z(1)), Z(2) * std::cos(Z(1)) * std::cos(Z(0))};
            const auto residual = predictor_single::residual(*selected_observation, predicted_observation);
            prediction.errors << plate(0) - observed_position(0), plate(2) - observed_position(2), -residual(0), -residual(2); // 计算残差
        }
        prediction.status = updated ? "updated" : "prediction_only";
    }

    if (is_initialized)
    {
        auto future = X;
        for (int i : {0, 2, 4, 6})
            future(i) += horizon_ms / 1000 * future(i + 1);
        future(6) = predictor_single::wrap_to_pi(future(6));
        prediction.current.plates.push_back(predictor_single::plate_pose(X));
        prediction.future.plates.push_back(predictor_single::plate_pose(future));
        if (prediction.status != "initialized")
        {
            prediction.values = {plate(0), observed_position(0), plate(2), observed_position(2), predicted_observation(0),
                                 selected_observation ? (*selected_observation)(0) : missing, predicted_observation(2),
                                 selected_observation ? (*selected_observation)(2) : missing};
        }
    }
    return prediction;
}