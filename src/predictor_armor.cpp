#include "predictor_armor.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Fixed observation standard deviations: [0.04 rad, 0.04 rad, 0.40 m, 0.24 rad].

double armor_wrap_to_pi(double a) {
    double r = std::fmod(a + armor_pi, 2 * armor_pi);
    if (r < 0)
        r += 2 * armor_pi;
    return r - armor_pi;
}

ArmorMeasurement armor_angular_residual(const ArmorMeasurement &a, const ArmorMeasurement &b) {
    ArmorMeasurement d = a - b;
    for (int i : {0, 1, 3})
        d(i) = armor_wrap_to_pi(d(i));
    return d;
}

std::pair<ArmorCovariance, ArmorCovariance> armor_transition(const ArmorEKF &self, double dt) {
    if (!std::isfinite(dt) || dt < 0)
        throw std::invalid_argument("dt must be finite and nonnegative");
    ArmorCovariance f = ArmorCovariance::Identity(), q = ArmorCovariance::Zero();
    for (int i : {0, 2, 4, 6}) {
        f(i, i + 1) = dt;
        double noise = i == 6 ? self.q_yaw : self.q_pos;
        q(i, i) = dt * dt * dt / 3 * noise;
        q(i, i + 1) = q(i + 1, i) = dt * dt / 2 * noise;
        q(i + 1, i + 1) = dt * noise;
    }
    q(8, 8) = self.q_r * dt;
    q(9, 9) = self.q_dl * dt;
    q(10, 10) = self.q_dh * dt;
    return {f, q};
}

void armor_predict(ArmorEKF &self, double dt) {
    auto [f, q] = armor_transition(self, dt);
    self.F = f;
    self.X = self.F * self.X;
    self.X(6) = armor_wrap_to_pi(self.X(6));
    self.P = self.F * self.P * self.F.transpose() + q;
}

Forecast armor_forecast(const ArmorEKF &self, double horizon_s) {
    if (!self.is_initialized)
        throw std::invalid_argument("Cannot forecast before initialization");
    auto [f, q] = armor_transition(self, horizon_s);
    ArmorState s = f * self.X;
    s(6) = armor_wrap_to_pi(s(6));
    Eigen::Matrix4d poses;
    for (int id = 0; id < 4; ++id)
        poses.row(id) = armor_plate_pose(s, id).transpose();
    return {s, f * self.P * f.transpose() + q, poses};
}

ArmorMeasurement armor_plate_pose(const ArmorState &s, int id) {
    double yaw = s(6) + id * armor_pi / 2, r = s(8) + (id % 2 ? s(9) : 0),
           x = s(0) + r * std::sin(yaw), y = s(2) + (id % 2 ? s(10) : 0),
           z = s(4) - r * std::cos(yaw);
    return {x, y, z, armor_wrap_to_pi(yaw)};
}

ArmorMeasurement armor_h(const ArmorState &s, int id) {
    const auto pose = armor_plate_pose(s, id);
    const double x = pose(0), y = pose(1), z = pose(2);
    return {armor_wrap_to_pi(std::atan2(x, z)),
            armor_wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), pose(3)};
}

ArmorJacobian armor_get_jacobian(const ArmorState &s, int id) {
    ArmorJacobian H = ArmorJacobian::Zero();
    const ArmorMeasurement base = armor_h(s, id);
    for (int i : {0, 2, 4, 6, 8, 9, 10}) {
        ArmorState t = s;
        t(i) += 1e-5;
        const ArmorMeasurement d = armor_angular_residual(armor_h(t, id), base);
        H.col(i) = d / 1e-5;
    }
    return H;
}

bool armor_valid_observation(const ArmorEKF &self, const ArmorMeasurement &z) {
    return z.allFinite() && z(2) > 0 &&
           std::abs(z(1)) < armor_pi / 2 && (self.base_frame || std::abs(z(0)) < armor_pi / 2);
}

void armor_initialize(ArmorEKF &self, const ArmorMeasurement &z) {
    if (!armor_valid_observation(self, z))
        throw std::invalid_argument("Invalid initialization observation");
    double x = z(2) * std::cos(z(1)) * std::sin(z(0)), y = z(2) * std::sin(z(1)),
           zc = z(2) * std::cos(z(1)) * std::cos(z(0));
    self.X << x - .26 * std::sin(z(3)), 0, y, 0, zc + .26 * std::cos(z(3)), 0, z(3), 0, .26, 0, 0;
    self.P = (ArmorState() << .1, 1, .1, 1, .1, 1, .05, 100, .01, .01, .01).finished().asDiagonal();
    self.is_initialized = true;
}

Association armor_associate(const ArmorEKF &self, const std::vector<Observation> &observations) {
    std::vector<std::vector<Match>> candidates;
    std::vector<Diagnostic> diagnostics;
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto &o = observations[index];
        Diagnostic d;
        d.observation_index = index;
        std::vector<Match> options;
        if (armor_valid_observation(self, o.Z_obs)) {
            for (int id = 0; id < 4; ++id) {
                if (o.armor_id != -1 && o.armor_id != id)
                    continue;
                const ArmorMeasurement pred = armor_h(self.X, id), res = armor_angular_residual(o.Z_obs, pred);
                const ArmorJacobian H = armor_get_jacobian(self.X, id);
                const Eigen::Matrix4d S = H * self.P * H.transpose() + self.R;
                if (!S.allFinite())
                    throw std::runtime_error("Non-finite armor innovation covariance");
                const Eigen::LDLT<Eigen::Matrix4d> solver(S);
                if (solver.info() != Eigen::Success || solver.vectorD().minCoeff() <= 0)
                    throw std::runtime_error("Armor innovation covariance must be positive definite");
                const double nis = res.dot(solver.solve(res));
                if (!std::isfinite(nis))
                    throw std::runtime_error("Non-finite armor innovation");
                if (d.best_candidate_id < 0 || nis < d.nis) {
                    d.nis = nis;
                    d.best_candidate_id = id;
                    d.distance_residual = res(2);
                }
                if (nis <= self.nis_gate && std::abs(res(2)) <= self.max_distance_error)
                    options.push_back({index, id, nis, res, H, o.Z_obs});
            }
            d.reason = options.empty() ? "innovation_gate" : "association_conflict";
        }
        candidates.push_back(options);
        diagnostics.push_back(d);
    }
    std::vector<Match> best, selected;
    double best_score = std::numeric_limits<double>::infinity();
    auto search = [&](auto &&recurse, size_t index, unsigned used, double score) -> void {
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
                if (std::abs(armor_wrap_to_pi(c.Z_obs(3) - o.Z_obs(3) -
                                        (c.armor_id - o.armor_id) * armor_pi / 2)) >
                    self.pair_yaw_tolerance) {
                    consistent = false;
                    break;
                }
            if (consistent) {
                selected.push_back(c);
                recurse(recurse, index + 1, used | (1u << c.armor_id), score + c.nis);
                selected.pop_back();
            }
        }
        recurse(recurse, index + 1, used, score);
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

Association armor_update_multi(ArmorEKF &self, const std::vector<Observation> &obs) {
    auto result = armor_associate(self, obs);
    auto &matches = result.first;
    if (matches.empty())
        return result;
    int n = static_cast<int>(matches.size()) * 4;
    ArmorJointJacobian H(n, 11);
    ArmorJointResidual res(n);
    ArmorJointNoise noise = ArmorJointNoise::Zero(n, n);
    for (size_t k = 0; k < matches.size(); ++k) {
        const auto offset = static_cast<Eigen::Index>(4 * k);
        res.segment<4>(offset) = matches[k].residual;
        H.middleRows<4>(offset) = matches[k].H;
        noise.block<4, 4>(offset, offset) = self.R;
    }
    const ArmorJointNoise S = H * self.P * H.transpose() + noise;
    if (!S.allFinite())
        throw std::runtime_error("Non-finite armor joint covariance");
    const Eigen::LDLT<ArmorJointNoise> solver(S);
    if (solver.info() != Eigen::Success || solver.vectorD().minCoeff() <= 0)
        throw std::runtime_error("Armor joint covariance must be positive definite");
    const ArmorGain K = solver.solve(H * self.P).transpose();
    if (!K.allFinite())
        throw std::runtime_error("Non-finite armor Kalman gain");
    self.X = self.X + K * res;
    self.X(6) = armor_wrap_to_pi(self.X(6));
    self.X(8) = std::clamp(self.X(8), .20, .30);
    self.X(9) = std::clamp(self.X(8) + self.X(9), .20, .30) - self.X(8);
    const ArmorCovariance A = ArmorCovariance::Identity() - K * H;
    self.P = A * self.P * A.transpose() + K * noise * K.transpose();
    self.P = ((self.P + self.P.transpose()) * .5).eval();
    return result;
}
