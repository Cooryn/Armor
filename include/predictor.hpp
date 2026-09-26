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
constexpr double FPS = 30.0;
class BasicPredictor {
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
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6), H = Eigen::MatrixXd::Zero(3, 6), P = Eigen::MatrixXd::Identity(6, 6) * 10,
           Q = Eigen::MatrixXd::Zero(6, 6), R = Eigen::MatrixXd::Identity(3, 3) * .1;
    double q = .01;
    bool is_initialized = false;
    BasicPredictor() {
        for (int i = 0; i < 3; ++i)
            H(i, 2 * i) = 1;
    }
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> predict(double dt) {
        Q = Eigen::MatrixXd::Zero(6, 6);
        for (int i : {0, 2, 4}) {
            F(i, i + 1) = dt;
            Q(i, i) = dt * dt * dt * dt / 4 * q;
            Q(i, i + 1) = Q(i + 1, i) = dt * dt * dt / 2 * q;
            Q(i + 1, i + 1) = dt * dt * q;
        }
        return {F * state, F * P * F.transpose() + Q};
    }
    void update(const Eigen::MatrixXd &Z, const Eigen::MatrixXd &predicted_state, const Eigen::MatrixXd &predicted_P) {
        Eigen::MatrixXd S = H * predicted_P * H.transpose() + R;
        Eigen::MatrixXd K = solve(S, H * predicted_P).transpose();
        state = predicted_state + K * (Z - H * predicted_state);
        P = (Eigen::MatrixXd::Identity(6, 6) - K * H) * predicted_P;
    }
};
} // namespace predictor
