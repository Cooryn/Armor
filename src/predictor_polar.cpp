#include "predictor_polar.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

double polar_wrap_to_pi(double a)
{
    double r = std::fmod(a + polar_pi, 2 * polar_pi);
    if (r < 0)
        r += 2 * polar_pi;
    return r - polar_pi;
}

Eigen::Vector4d polar_angular_residual(const Eigen::Vector4d &a, const Eigen::Vector4d &b)
{
    Eigen::Vector4d d = a - b;
    for (int i : {0, 1, 3})
        d(i) = polar_wrap_to_pi(d(i));
    return d;
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

Eigen::Vector4d polar_h(const Eigen::Matrix<double, 9, 1> &s, int plate_idx)
{
    double yaw = s(6) + plate_idx * polar_pi / 2, x = s(0) + s(8) * std::sin(yaw),
           z = s(4) - s(8) * std::cos(yaw), y = s(2);
    return {polar_wrap_to_pi(std::atan2(x, z)),
            polar_wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), polar_wrap_to_pi(yaw)};
}

Eigen::Matrix<double, 4, 9> polar_get_jacobian(const Eigen::Matrix<double, 9, 1> &s, int id)
{
    Eigen::Matrix<double, 4, 9> H = Eigen::Matrix<double, 4, 9>::Zero();
    const Eigen::Vector4d base = polar_h(s, id);
    for (int i = 0; i < 9; ++i)
    {
        Eigen::Matrix<double, 9, 1> t = s;
        t(i) += 1e-5;
        const Eigen::Vector4d d = polar_angular_residual(polar_h(t, id), base);
        H.col(i) = d / 1e-5;
    }
    return H;
}

void PolarEKF::update(const Eigen::Vector4d &z)
{
    int id = polar_round_even(polar_wrap_to_pi(z(3) - X(6)) / (polar_pi / 2));
    const Eigen::Matrix<double, 4, 9> H = polar_get_jacobian(X, id);
    const Eigen::Vector4d Y = polar_angular_residual(z, polar_h(X, id));
    const Eigen::Matrix4d S = H * P * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix4d> solver(S);
    const Eigen::Matrix<double, 9, 4> PHt = P * H.transpose();
    const Eigen::Matrix<double, 9, 4> K = solver.solve(PHt.transpose()).transpose();
    X = X + K * Y;
    X(6) = polar_wrap_to_pi(X(6));
    P = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * P;
}

VideoPrediction PolarEKF::update_frame(std::int64_t frame, double timestamp_ms,
                                      const std::vector<VideoObservation> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms_) / 1000;
    last_timestamp_ms_ = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    VideoPrediction result;
    result.horizon_ms = horizon_ms;
    const VideoObservation *selected = nullptr;
    for (const auto &observation : observations)
    {
        const auto &z = observation.measurement;
        if (z(2) > 1e-6 && std::abs(z(1)) < polar_pi / 2 && z(2) * std::cos(z(1)) > 1e-6 &&
            (!selected || z(2) < selected->measurement(2)))
            selected = &observation;
    }
    if (!is_initialized)
    {
        if (selected)
        {
            const auto &p = selected->position;
            const auto &z = selected->measurement;
            X << p(0) - .26 * std::sin(z(3)), 0, p(1), 0,
                 p(2) + .26 * std::cos(z(3)), 0, z(3), 0, .26;
            is_initialized = true;
        }
        result.status = selected ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        if (selected)
        {
            const int id = polar_round_even(polar_wrap_to_pi(selected->measurement(3) - X(6)) / (polar_pi / 2));
            result.errors = polar_angular_residual(polar_h(X, id), selected->measurement);
            update(selected->measurement);
        }
        const auto &errors = result.errors;
        result.values = {double(frame), X(0), X(1), X(2), X(3), X(4), X(5), X(6), X(7), X(8),
                         errors(0), errors(1), errors(2), errors(3), selected ? selected->measurement(3) : missing};
        result.value_count = 15;
        result.status = selected ? "updated" : "prediction_only";
    }
    result.initialized = is_initialized;
    if (is_initialized)
    {
        result.position_variance = std::max({P(0, 0), P(2, 2), P(4, 4)});
        auto future = X;
        for (int i : {0, 2, 4, 6})
            future(i) += horizon_ms / 1000 * future(i + 1);
        future(6) = polar_wrap_to_pi(future(6));
        for (bool forecast : {false, true})
        {
            const auto &s = forecast ? future : X;
            auto &geometry = forecast ? result.future : result.current;
            geometry.center = Eigen::Vector3d(s(0), s(2), s(4));
            for (int i = 0; i < 4; ++i)
            {
                const double yaw = polar_wrap_to_pi(s(6) + i * polar_pi / 2);
                geometry.plates.emplace_back(s(0) + s(8) * std::sin(yaw), s(2), s(4) - s(8) * std::cos(yaw), yaw);
            }
        }
    }
    return result;
}
