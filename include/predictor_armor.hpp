#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace predictor {
using ArmorState = Eigen::Matrix<double, 11, 1>;
using ArmorCovariance = Eigen::Matrix<double, 11, 11>;
using ArmorJacobian = Eigen::Matrix<double, 4, 11>;
struct Observation {
    Eigen::Vector4d Z_obs = Eigen::Vector4d::Zero();
    std::optional<int> armor_id = std::nullopt;
};
struct Match {
    size_t index;
    int armor_id;
    double nis;
    Eigen::Vector4d residual;
    ArmorJacobian H;
    Eigen::Vector4d predicted, Z_obs;
};
struct Diagnostic {
    size_t observation_index;
    bool accepted = false;
    int armor_id = -1;
    double nis = std::numeric_limits<double>::quiet_NaN();
    std::string reason = "invalid";
    int best_candidate_id = -1;
    double distance_residual = std::numeric_limits<double>::quiet_NaN();
};
struct Forecast {
    ArmorState state;
    ArmorCovariance covariance;
    Eigen::Matrix4d plates;
};
struct Innovation {
    double nis;
    Eigen::Vector4d residual;
    ArmorJacobian H;
    Eigen::Vector4d prediction;
};
using Association = std::pair<std::vector<Match>, std::vector<Diagnostic>>;
class ArmorEKF {
  public:
    using State = ArmorState;
    using Covariance = ArmorCovariance;
    using Measurement = Eigen::Vector4d;
    using Jacobian = ArmorJacobian;
    // One to four accepted plates, four observation components per plate.
    using JointJacobian = Eigen::Matrix<double, Eigen::Dynamic, 11, 0, 16, 11>;
    using JointResidual = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 16, 1>;
    using JointNoise = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16>;
    using Gain = Eigen::Matrix<double, 11, Eigen::Dynamic, 0, 11, 16>;
    static constexpr double pi = 3.14159265358979323846;
    static double wrap_to_pi(double a) {
        double r = std::fmod(a + pi, 2 * pi);
        if (r < 0)
            r += 2 * pi;
        return r - pi;
    }
    static Measurement angular_residual(const Measurement &a, const Measurement &b) {
        Measurement d = a - b;
        for (int i : {0, 1, 3})
            d(i) = wrap_to_pi(d(i));
        return d;
    }
    State X = State::Zero();
    // Fixed variances for [target yaw, target pitch, distance, plate yaw].
    // Effective standard deviations: [0.04 rad, 0.04 rad, 0.40 m, 0.24 rad].
    Covariance F = Covariance::Identity(), P = Covariance::Identity() * 10;
    Eigen::Matrix4d R = Measurement(.0016, .0016, .16, .0576).asDiagonal();
    double q_pos = 3, q_yaw = 15, q_r = 3e-4, q_dl = 3e-3, q_dh = 3e-3;
    bool is_initialized = false;
    double nis_gate, pair_yaw_tolerance, max_distance_error;
    explicit ArmorEKF(double gate = 16, double tolerance = 25 * pi / 180,
                      double distance_error = .5)
        : nis_gate(gate), pair_yaw_tolerance(tolerance),
          max_distance_error(distance_error) {
        P(8, 8) = .01;
        P(9, 9) = P(10, 10) = .05;
    }
    std::pair<Covariance, Covariance> _transition(double dt) const {
        if (!std::isfinite(dt) || dt < 0)
            throw std::invalid_argument("dt must be finite and nonnegative");
        Covariance f = Covariance::Identity(), q = Covariance::Zero();
        for (int i : {0, 2, 4, 6}) {
            f(i, i + 1) = dt;
            double noise = i == 6 ? q_yaw : q_pos;
            q(i, i) = dt * dt * dt / 3 * noise;
            q(i, i + 1) = q(i + 1, i) = dt * dt / 2 * noise;
            q(i + 1, i + 1) = dt * noise;
        }
        q(8, 8) = q_r * dt;
        q(9, 9) = q_dl * dt;
        q(10, 10) = q_dh * dt;
        return {f, q};
    }
    void predict(double dt) {
        auto [f, q] = _transition(dt);
        F = f;
        X = F * X;
        X(6) = wrap_to_pi(X(6));
        P = F * P * F.transpose() + q;
    }
    Forecast forecast(double horizon_s = .05) const {
        if (!is_initialized)
            throw std::invalid_argument("Cannot forecast before initialization");
        auto [f, q] = _transition(horizon_s);
        State s = f * X;
        s(6) = wrap_to_pi(s(6));
        Eigen::Matrix4d poses;
        for (int id = 0; id < 4; ++id)
            poses.row(id) = plate_pose(s, id).transpose();
        return {s, f * P * f.transpose() + q, poses};
    }
    Measurement plate_pose(const State &s, int id) const {
        double yaw = s(6) + id * pi / 2, r = s(8) + (id % 2 ? s(9) : 0),
               x = s(0) + r * std::sin(yaw), y = s(2) + (id % 2 ? s(10) : 0),
               z = s(4) - r * std::cos(yaw);
        return {x, y, z, wrap_to_pi(yaw)};
    }
    Measurement h(const State &s, int id) const {
        const auto pose = plate_pose(s, id);
        const double x = pose(0), y = pose(1), z = pose(2);
        return {wrap_to_pi(std::atan2(x, z)),
                               wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
                               std::sqrt(x * x + y * y + z * z), pose(3)};
    }
    Jacobian get_jacobian(const State &s, int id) const {
        Jacobian H = Jacobian::Zero();
        const Measurement base = h(s, id);
        for (int i : {0, 2, 4, 6, 8, 9, 10}) {
            State t = s;
            t(i) += 1e-5;
            const Measurement d = angular_residual(h(t, id), base);
            H.col(i) = d / 1e-5;
        }
        return H;
    }
    bool base_frame = false;
    bool valid_observation(const Measurement &z) const {
        return z.allFinite() && z(2) > 0 &&
               std::abs(z(1)) < pi / 2 && (base_frame || std::abs(z(0)) < pi / 2);
    }
    void initialize(const Measurement &z) {
        if (!valid_observation(z))
            throw std::invalid_argument("Invalid initialization observation");
        double x = z(2) * std::cos(z(1)) * std::sin(z(0)), y = z(2) * std::sin(z(1)),
               zc = z(2) * std::cos(z(1)) * std::cos(z(0));
        X << x - .26 * std::sin(z(3)), 0, y, 0, zc + .26 * std::cos(z(3)), 0, z(3), 0, .26, 0, 0;
        P = (State() << .1, 1, .1, 1, .1, 1, .05, 100, .01, .01, .01).finished().asDiagonal();
        is_initialized = true;
    }
    Innovation innovation(const Measurement &z, int id) const {
        const Measurement pred = h(X, id), res = angular_residual(z, pred);
        const Jacobian H = get_jacobian(X, id);
        const Eigen::Matrix4d S = H * P * H.transpose() + R;
        if (!S.allFinite())
            throw std::runtime_error("Non-finite armor innovation covariance");
        const Eigen::LDLT<Eigen::Matrix4d> solver(S);
        if (solver.info() != Eigen::Success || solver.vectorD().minCoeff() <= 0)
            throw std::runtime_error("Armor innovation covariance must be positive definite");
        const double nis = res.dot(solver.solve(res));
        if (!std::isfinite(nis))
            throw std::runtime_error("Non-finite armor innovation");
        return {nis, res, H, pred};
    }
    Association associate(const std::vector<Observation> &observations) const {
        std::vector<std::vector<Match>> candidates;
        std::vector<Diagnostic> diagnostics;
        for (size_t index = 0; index < observations.size(); ++index) {
            const auto &o = observations[index];
            Diagnostic d;
            d.observation_index = index;
            std::vector<Match> options;
            if (valid_observation(o.Z_obs)) {
                for (int id = 0; id < 4; ++id) {
                    if (o.armor_id && *o.armor_id != id)
                        continue;
                    auto in = innovation(o.Z_obs, id);
                    if (d.best_candidate_id < 0 || in.nis < d.nis) {
                        d.nis = in.nis;
                        d.best_candidate_id = id;
                        d.distance_residual = in.residual(2);
                    }
                    if (in.nis <= nis_gate && std::abs(in.residual(2)) <= max_distance_error)
                        options.push_back({index, id, in.nis, in.residual, in.H, in.prediction,
                                           o.Z_obs});
                }
                d.reason = options.empty() ? "innovation_gate" : "association_conflict";
            }
            candidates.push_back(options);
            diagnostics.push_back(d);
        }
        std::vector<Match> best, selected;
        double best_score = std::numeric_limits<double>::infinity();
        auto search = [&](auto &&self, size_t index, unsigned used, double score) -> void {
            if (selected.size() + std::min(size_t(4) - selected.size(), candidates.size() - index) <
                best.size())
                return;
            if (index == candidates.size()) {
                if (selected.size() > best.size() ||
                    (selected.size() == best.size() && score < best_score)) {
                    best = selected;
                    best_score = score;
                }
                return;
            }
            for (const auto &c : candidates[index]) {
                if (used & (1u << c.armor_id))
                    continue;
                bool consistent = true;
                for (const auto &o : selected)
                    if (std::abs(wrap_to_pi(c.Z_obs(3) - o.Z_obs(3) -
                                            (c.armor_id - o.armor_id) * pi / 2)) >
                        pair_yaw_tolerance) {
                        consistent = false;
                        break;
                    }
                if (consistent) {
                    selected.push_back(c);
                    self(self, index + 1, used | (1u << c.armor_id), score + c.nis);
                    selected.pop_back();
                }
            }
            self(self, index + 1, used, score);
        };
        search(search, 0, 0, 0);
        for (const auto &m : best) {
            auto &d = diagnostics[m.index];
            d.accepted = true;
            d.armor_id = m.armor_id;
            d.nis = m.nis;
            d.reason = "accepted";
            d.distance_residual = m.residual(2);
        }
        return {best, diagnostics};
    }
    Association update_multi(const std::vector<Observation> &obs) {
        auto result = associate(obs);
        auto &matches = result.first;
        if (matches.empty())
            return result;
        int n = static_cast<int>(matches.size()) * 4;
        JointJacobian H(n, 11);
        JointResidual res(n);
        JointNoise noise = JointNoise::Zero(n, n);
        for (size_t k = 0; k < matches.size(); ++k) {
            const auto offset = static_cast<Eigen::Index>(4 * k);
            res.segment<4>(offset) = matches[k].residual;
            H.middleRows<4>(offset) = matches[k].H;
            noise.block<4, 4>(offset, offset) = R;
        }
        const JointNoise S = H * P * H.transpose() + noise;
        if (!S.allFinite())
            throw std::runtime_error("Non-finite armor joint covariance");
        const Eigen::LDLT<JointNoise> solver(S);
        if (solver.info() != Eigen::Success || solver.vectorD().minCoeff() <= 0)
            throw std::runtime_error("Armor joint covariance must be positive definite");
        const Gain K = solver.solve(H * P).transpose();
        if (!K.allFinite())
            throw std::runtime_error("Non-finite armor Kalman gain");
        X = X + K * res;
        X(6) = wrap_to_pi(X(6));
        X(8) = std::clamp(X(8), .20, .30);
        X(9) = std::clamp(X(8) + X(9), .20, .30) - X(8);
        const Covariance A = Covariance::Identity() - K * H;
        P = A * P * A.transpose() + K * noise * K.transpose();
        P = ((P + P.transpose()) * .5).eval();
        return result;
    }
    std::optional<int> update(const Measurement &z, std::optional<int> id = std::nullopt) {
        auto result = update_multi({Observation{z, id}});
        if (result.first.empty())
            return std::nullopt;
        return result.first[0].armor_id;
    }
};
} // namespace predictor
