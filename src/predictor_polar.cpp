#include "predictor_polar.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

double polar_wrap_to_pi(double a) {
    double r = std::fmod(a + polar_pi, 2 * polar_pi);
    if (r < 0)
        r += 2 * polar_pi;
    return r - polar_pi;
}

Eigen::Vector4d polar_angular_residual(const Eigen::Vector4d &a, const Eigen::Vector4d &b) {
    Eigen::Vector4d d = a - b;
    for (int i : {0, 1, 3})
        d(i) = polar_wrap_to_pi(d(i));
    return d;
}

int polar_round_even(double x) {
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

Eigen::Vector4d polar_h(const Eigen::Matrix<double, 9, 1> &s, int plate_idx) {
    double yaw = s(6) + plate_idx * polar_pi / 2, x = s(0) + s(8) * std::sin(yaw),
           z = s(4) - s(8) * std::cos(yaw), y = s(2);
    return {polar_wrap_to_pi(std::atan2(x, z)),
            polar_wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), polar_wrap_to_pi(yaw)};
}

Eigen::Matrix<double, 4, 9> polar_get_jacobian(const Eigen::Matrix<double, 9, 1> &s, int id) {
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
    if (!S.allFinite() || !Y.allFinite())
        throw std::runtime_error("Non-finite polar innovation");
    const Eigen::LDLT<Eigen::Matrix4d> solver(S);
    if (solver.info() != Eigen::Success || solver.vectorD().minCoeff() <= 0)
        throw std::runtime_error("Polar innovation covariance must be positive definite");
    const Eigen::Matrix<double, 9, 4> PHt = P * H.transpose();
    const Eigen::Matrix<double, 9, 4> K = solver.solve(PHt.transpose()).transpose();
    if (!K.allFinite())
        throw std::runtime_error("Non-finite polar Kalman gain");
    X = X + K * Y;
    X(6) = polar_wrap_to_pi(X(6));
    P = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * P;
}
