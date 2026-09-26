#pragma once
#include <Eigen/Dense>
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace predictor {
class PolarEKF {
  public:
    static constexpr double pi = 3.14159265358979323846;
    static double wrap_to_pi(double a) {
        double r = std::fmod(a + pi, 2 * pi);
        if (r < 0)
            r += 2 * pi;
        return r - pi;
    }
    static Eigen::VectorXd make_vector(std::initializer_list<double> values) {
        return Eigen::Map<const Eigen::VectorXd>(values.begin(), static_cast<Eigen::Index>(values.size()));
    }
    static Eigen::MatrixXd diagonal(std::initializer_list<double> values) {
        return make_vector(values).asDiagonal();
    }
    static Eigen::MatrixXd solve(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
        if (a.rows() != a.cols() || a.rows() != b.rows() || a.rows() == 0)
            throw std::invalid_argument("Solve shape mismatch");
        if (!a.allFinite() || !b.allFinite())
            throw std::invalid_argument("Non-finite linear system");
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(a);
        if ((lu.matrixLU().diagonal().array() == 0).any())
            throw std::runtime_error("Singular matrix");
        Eigen::MatrixXd result = lu.solve(b);
        if (!result.allFinite())
            throw std::runtime_error("Non-finite linear solution");
        return result;
    }
    static Eigen::MatrixXd angular_residual(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
        if (a.rows() != 4 || a.cols() != 1 || b.rows() != 4 || b.cols() != 1)
            throw std::invalid_argument("Observation shape mismatch");
        Eigen::MatrixXd d = a - b;
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
    Eigen::VectorXd X = Eigen::VectorXd::Zero(9);
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(9, 9), P = Eigen::MatrixXd::Identity(9, 9) * 10, Q = Eigen::MatrixXd::Identity(9, 9) * .01,
           R = diagonal({.005, .005, .05, .05});
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
    Eigen::MatrixXd h(const Eigen::MatrixXd &s, int plate_idx) const {
        double yaw = s(6) + plate_idx * pi / 2, x = s(0) - s(8) * std::sin(yaw),
               z = s(4) - s(8) * std::cos(yaw), y = s(2);
        return make_vector({wrap_to_pi(std::atan2(x, z)),
                               wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
                               std::sqrt(x * x + y * y + z * z), wrap_to_pi(yaw)});
    }
    Eigen::MatrixXd get_jacobian(const Eigen::MatrixXd &s, int id) const {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 9), base = h(s, id);
        for (int i = 0; i < 9; ++i) {
            Eigen::MatrixXd t = s;
            t(i) += 1e-5;
            Eigen::MatrixXd d = angular_residual(h(t, id), base);
            H.col(i) = d / 1e-5;
        }
        return H;
    }
    void update(const Eigen::MatrixXd &z) {
        int id = round_even(wrap_to_pi(z(3) - X(6)) / (pi / 2));
        Eigen::MatrixXd H = get_jacobian(X, id), Y = angular_residual(z, h(X, id)), S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = (P * H.transpose()) * solve(S, Eigen::MatrixXd::Identity(4, 4));
        X = X + K * Y;
        X(6) = wrap_to_pi(X(6));
        P = (Eigen::MatrixXd::Identity(9, 9) - K * H) * P;
    }
};
} // namespace predictor
