#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace predictor {
class SinglePlateEKF {
  public:
    using State = Eigen::Matrix<double, 6, 1>; // [x, vx, y, vy, z, vz]
    using Covariance = Eigen::Matrix<double, 6, 6>;
    using Observation = Eigen::Vector3d; // [yaw (rad), pitch (rad), distance (m)]
    using Jacobian = Eigen::Matrix<double, 3, 6>;
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double geometry_epsilon = 1e-6;
    static constexpr double process_noise = .01;

    static double wrap_to_pi(double angle) {
        double result = std::fmod(angle + pi, 2 * pi);
        return (result < 0 ? result + 2 * pi : result) - pi;
    }
    static Eigen::Matrix3d observation_noise() {
        return Observation(.0016, .0016, .16).asDiagonal();
    }
    static bool valid_observation(const Observation &z) {
        return z.allFinite() && z(2) > geometry_epsilon && std::abs(z(1)) < pi / 2 &&
               z(2) * std::cos(z(1)) > geometry_epsilon;
    }
    static Observation h(const State &s) {
        const double horizontal = std::hypot(s(0), s(4));
        return {std::atan2(s(0), s(4)), std::atan2(s(2), horizontal), std::hypot(horizontal, s(2))};
    }
    static Jacobian jacobian(const State &s) {
        if (!valid_geometry(s))
            throw std::invalid_argument("Singular single-plate observation geometry");
        const double x = s(0), y = s(2), z = s(4), horizontal = std::hypot(x, z),
                     distance = std::hypot(horizontal, y), horizontal2 = horizontal * horizontal,
                     distance2 = distance * distance;
        Jacobian H = Jacobian::Zero();
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
    void initialize(const Observation &z) {
        if (!valid_observation(z))
            throw std::invalid_argument("Invalid single-plate initialization observation");
        const double horizontal = z(2) * std::cos(z(1));
        state_ << horizontal * std::sin(z(0)), 0, z(2) * std::sin(z(1)), 0,
                  horizontal * std::cos(z(0)), 0;
        covariance_ = Covariance::Identity() * 10;
        initialized_ = true;
    }
    void predict(double dt) {
        if (!initialized_ || !std::isfinite(dt) || dt < 0)
            throw std::invalid_argument("Prediction requires initialization and finite nonnegative dt");
        Covariance F = Covariance::Identity(), Q = Covariance::Zero();
        const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
        for (int i = 0; i < 6; i += 2) {
            F(i, i + 1) = dt;
            Q(i, i) = process_noise * dt4 / 4;
            Q(i, i + 1) = Q(i + 1, i) = process_noise * dt3 / 2;
            Q(i + 1, i + 1) = process_noise * dt2;
        }
        const State predicted = F * state_;
        const Covariance covariance = F * covariance_ * F.transpose() + Q;
        if (!predicted.allFinite() || !covariance.allFinite())
            throw std::invalid_argument("Non-finite single-plate prediction");
        state_ = predicted;
        covariance_ = covariance;
    }
    bool update(const Observation &z) {
        if (!initialized_ || !valid_observation(z) || !valid_geometry(state_))
            return false;
        const Jacobian H = jacobian(state_);
        Observation residual = z - h(state_);
        residual(0) = wrap_to_pi(residual(0));
        residual(1) = wrap_to_pi(residual(1));
        const Eigen::Matrix3d R = observation_noise(), S = H * covariance_ * H.transpose() + R;
        const Eigen::LDLT<Eigen::Matrix3d> solver(S);
        if (solver.info() != Eigen::Success || !solver.isPositive())
            return false;
        const Eigen::Matrix<double, 6, 3> K = solver.solve(H * covariance_).transpose();
        const State corrected = state_ + K * residual;
        const Covariance A = Covariance::Identity() - K * H;
        Covariance covariance = A * covariance_ * A.transpose() + K * R * K.transpose();
        covariance = ((covariance + covariance.transpose()) * .5).eval();
        if (!corrected.allFinite() || !covariance.allFinite())
            return false;
        state_ = corrected;
        covariance_ = covariance;
        return true;
    }
    const State &state() const { return state_; }
    const Covariance &covariance() const { return covariance_; }
    bool initialized() const { return initialized_; }

  private:
    static bool valid_geometry(const State &s) {
        return s.allFinite() && std::hypot(s(0), s(4)) > geometry_epsilon &&
               std::hypot(std::hypot(s(0), s(4)), s(2)) > geometry_epsilon;
    }
    State state_ = State::Zero();
    Covariance covariance_ = Covariance::Identity() * 10;
    bool initialized_ = false;
};
} // namespace predictor
