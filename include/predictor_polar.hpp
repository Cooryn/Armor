#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
namespace predictor {
class PolarEKF {
  public:
    using State = Eigen::Matrix<double, 9, 1>;
    using Covariance = Eigen::Matrix<double, 9, 9>;
    using Observation = Eigen::Vector4d;
    using Jacobian = Eigen::Matrix<double, 4, 9>;
    using Gain = Eigen::Matrix<double, 9, 4>;
    static constexpr double pi = 3.14159265358979323846;
    static double wrap_to_pi(double a) {
        double r = std::fmod(a + pi, 2 * pi);
        if (r < 0)
            r += 2 * pi;
        return r - pi;
    }
    static Observation angular_residual(const Observation &a, const Observation &b) {
        Observation d = a - b;
        for (int i : {0, 1, 3})
            d(i) = wrap_to_pi(d(i));
        return d;
    }
    static int round_even(double x) {
        double lo = std::floor(x), f = x - lo;
        return static_cast<int>(f < .5                            ? lo
                                : f > .5                          ? lo + 1
                                : std::fmod(std::abs(lo), 2) == 0 ? lo
                                                                  : lo + 1);
    }
    State X = State::Zero();
    Covariance F = Covariance::Identity(), P = Covariance::Identity() * 10,
               Q = Covariance::Identity() * .01;
    Eigen::Matrix4d R = Observation(.005, .005, .05, .05).asDiagonal();
    bool is_initialized = false;
    PolarEKF() {
        P(8, 8) = .01;
        Q(1, 1) = Q(3, 3) = Q(5, 5) = .1;
        Q(7, 7) = .5;
        Q(8, 8) = .0001;
    }
    void predict(double dt) {
        for (int i : {0, 2, 4, 6})
            F(i, i + 1) = dt;
        X = F * X;
        X(6) = wrap_to_pi(X(6));
        P = F * P * F.transpose() + Q;
    }
    Observation h(const State &s, int plate_idx) const {
        double yaw = s(6) + plate_idx * pi / 2, x = s(0) - s(8) * std::sin(yaw),
               z = s(4) - s(8) * std::cos(yaw), y = s(2);
        return {wrap_to_pi(std::atan2(x, z)),
                               wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
                               std::sqrt(x * x + y * y + z * z), wrap_to_pi(yaw)};
    }
    Jacobian get_jacobian(const State &s, int id) const {
        Jacobian H = Jacobian::Zero();
        const Observation base = h(s, id);
        for (int i = 0; i < 9; ++i) {
            State t = s;
            t(i) += 1e-5;
            const Observation d = angular_residual(h(t, id), base);
            H.col(i) = d / 1e-5;
        }
        return H;
    }
    void update(const Observation &z) {
        int id = round_even(wrap_to_pi(z(3) - X(6)) / (pi / 2));
        const Jacobian H = get_jacobian(X, id);
        const Observation Y = angular_residual(z, h(X, id));
        const Eigen::Matrix4d S = H * P * H.transpose() + R;
        if (!S.allFinite() || !Y.allFinite())
            throw std::runtime_error("Non-finite polar innovation");
        const Eigen::LDLT<Eigen::Matrix4d> solver(S);
        if (solver.info() != Eigen::Success || solver.vectorD().minCoeff() <= 0)
            throw std::runtime_error("Polar innovation covariance must be positive definite");
        const Gain PHt = P * H.transpose();
        const Gain K = solver.solve(PHt.transpose()).transpose();
        if (!K.allFinite())
            throw std::runtime_error("Non-finite polar Kalman gain");
        X = X + K * Y;
        X(6) = wrap_to_pi(X(6));
        P = (Covariance::Identity() - K * H) * P;
    }
};
} // namespace predictor
