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

PolarObservation polar_angular_residual(const PolarObservation &a, const PolarObservation &b) {
    PolarObservation d = a - b;
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

void polar_predict(PolarEKF &self, double dt) {
    for (int i : {0, 2, 4, 6})
        self.F(i, i + 1) = dt;
    self.X = self.F * self.X;
    self.X(6) = polar_wrap_to_pi(self.X(6));
    self.P = self.F * self.P * self.F.transpose() + self.Q;
}

PolarObservation polar_h(const PolarState &s, int plate_idx) {
    double yaw = s(6) + plate_idx * polar_pi / 2, x = s(0) + s(8) * std::sin(yaw),
           z = s(4) - s(8) * std::cos(yaw), y = s(2);
    return {polar_wrap_to_pi(std::atan2(x, z)),
            polar_wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), polar_wrap_to_pi(yaw)};
}

PolarJacobian polar_get_jacobian(const PolarState &s, int id) {
    PolarJacobian H = PolarJacobian::Zero();
    const PolarObservation base = polar_h(s, id);
    for (int i = 0; i < 9; ++i)
    {
        PolarState t = s;
        t(i) += 1e-5;
        const PolarObservation d = polar_angular_residual(polar_h(t, id), base);
        H.col(i) = d / 1e-5;
    }
    return H;
}

void polar_update(PolarEKF &self, const PolarObservation &z) {
    int id = polar_round_even(polar_wrap_to_pi(z(3) - self.X(6)) / (polar_pi / 2));
    const PolarJacobian H = polar_get_jacobian(self.X, id);
    const PolarObservation Y = polar_angular_residual(z, polar_h(self.X, id));
    const Eigen::Matrix4d S = H * self.P * H.transpose() + self.R;
    if (!S.allFinite() || !Y.allFinite())
        throw std::runtime_error("Non-finite polar innovation");
    const Eigen::LDLT<Eigen::Matrix4d> solver(S);
    if (solver.info() != Eigen::Success || solver.vectorD().minCoeff() <= 0)
        throw std::runtime_error("Polar innovation covariance must be positive definite");
    const PolarGain PHt = self.P * H.transpose();
    const PolarGain K = solver.solve(PHt.transpose()).transpose();
    if (!K.allFinite())
        throw std::runtime_error("Non-finite polar Kalman gain");
    self.X = self.X + K * Y;
    self.X(6) = polar_wrap_to_pi(self.X(6));
    self.P = (PolarCovariance::Identity() - K * H) * self.P;
}
