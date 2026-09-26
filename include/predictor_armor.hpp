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
struct Observation {
    Eigen::MatrixXd Z_obs;
    std::optional<int> armor_id = std::nullopt;
    double detection_score = std::numeric_limits<double>::quiet_NaN(), reprojection_error = std::numeric_limits<double>::quiet_NaN();
};
struct Match {
    size_t index;
    int armor_id;
    double nis;
    Eigen::MatrixXd residual, H, predicted, Z_obs, R;
    double association_cost;
};
struct Diagnostic {
    size_t observation_index;
    bool accepted = false;
    int armor_id = -1;
    double nis = std::numeric_limits<double>::quiet_NaN();
    std::string reason = "invalid";
    int best_candidate_id = -1;
    double distance_residual = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd scales = Eigen::MatrixXd::Constant(4, 1, std::numeric_limits<double>::quiet_NaN());
};
struct Forecast {
    Eigen::MatrixXd state, covariance, plates;
};
struct Innovation {
    double nis;
    Eigen::MatrixXd residual, H, prediction;
};
using Association = std::pair<std::vector<Match>, std::vector<Diagnostic>>;
class ArmorEKF {
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
    static double logabsdet(const Eigen::MatrixXd &a) {
        if (a.rows() != a.cols() || a.rows() == 0 || !a.allFinite())
            throw std::invalid_argument("Invalid determinant matrix");
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(a);
        return lu.matrixLU().diagonal().array().abs().log().sum();
    }
    static Eigen::MatrixXd angular_residual(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
        if (a.rows() != 4 || a.cols() != 1 || b.rows() != 4 || b.cols() != 1)
            throw std::invalid_argument("Observation shape mismatch");
        Eigen::MatrixXd d = a - b;
        for (int i : {0, 1, 3})
            d(i) = wrap_to_pi(d(i));
        return d;
    }
    Eigen::VectorXd X = Eigen::VectorXd::Zero(11);
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11), P = Eigen::MatrixXd::Identity(11, 11) * 10,
           R = diagonal({.005, .005, .05, .05});
    double q_pos = 3, q_yaw = 15, q_r = 3e-4, q_dl = 3e-3, q_dh = 3e-3;
    bool adaptive_noise, is_initialized = false;
    double nis_gate, pair_yaw_tolerance, max_distance_error;
    explicit ArmorEKF(double gate = 16, double tolerance = 25 * pi / 180,
                      double distance_error = .5, bool adaptive = true)
        : adaptive_noise(adaptive), nis_gate(gate), pair_yaw_tolerance(tolerance),
          max_distance_error(distance_error) {
        P(8, 8) = .01;
        P(9, 9) = P(10, 10) = .05;
    }
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _transition(double dt) const {
        if (!std::isfinite(dt) || dt < 0)
            throw std::invalid_argument("dt must be finite and nonnegative");
        Eigen::MatrixXd f = Eigen::MatrixXd::Identity(11, 11), q = Eigen::MatrixXd::Zero(11, 11);
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
        Eigen::MatrixXd s = f * X;
        s(6) = wrap_to_pi(s(6));
        Eigen::MatrixXd poses(4, 4);
        for (int id = 0; id < 4; ++id) {
            double yaw = wrap_to_pi(s(6) + id * pi / 2), r = s(8) + (id % 2 ? s(9) : 0);
            poses(id, 0) = s(0) + r * std::sin(yaw);
            poses(id, 1) = s(2) + (id % 2 ? s(10) : 0);
            poses(id, 2) = s(4) - r * std::cos(yaw);
            poses(id, 3) = yaw;
        }
        return {s, f * P * f.transpose() + q, poses};
    }
    Eigen::MatrixXd h(const Eigen::MatrixXd &s, int id) const {
        double yaw = s(6) + id * pi / 2, r = s(8) + (id % 2 ? s(9) : 0),
               x = s(0) + r * std::sin(yaw), y = s(2) + (id % 2 ? s(10) : 0),
               z = s(4) - r * std::cos(yaw);
        return make_vector({wrap_to_pi(std::atan2(x, z)),
                               wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
                               std::sqrt(x * x + y * y + z * z), wrap_to_pi(yaw)});
    }
    Eigen::MatrixXd get_jacobian(const Eigen::MatrixXd &s, int id) const {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 11), base = h(s, id);
        for (int i = 0; i < 11; ++i) {
            Eigen::MatrixXd t = s;
            t(i) += 1e-5;
            Eigen::MatrixXd d = residual(h(t, id), base);
            H.col(i) = d / 1e-5;
        }
        return H;
    }
    static Eigen::MatrixXd residual(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
        return angular_residual(a, b);
    }
    bool base_frame = false;
    bool valid_observation(const Eigen::MatrixXd &z) const {
        if (z.rows() != 4 || z.cols() != 1)
            return false;
        if (!z.allFinite())
            return false;
        return z(2) > 0 && std::abs(z(1)) < pi / 2 && (base_frame || std::abs(z(0)) < pi / 2);
    }
    void initialize(const Eigen::MatrixXd &z) {
        if (!valid_observation(z))
            throw std::invalid_argument("Invalid initialization observation");
        double x = z(2) * std::cos(z(1)) * std::sin(z(0)), y = z(2) * std::sin(z(1)),
               zc = z(2) * std::cos(z(1)) * std::cos(z(0));
        X = make_vector(
            {x - .26 * std::sin(z(3)), 0, y, 0, zc + .26 * std::cos(z(3)), 0, z(3), 0, .26, 0, 0});
        P = diagonal({.1, 1, .1, 1, .1, 1, .05, 100, .01, .01, .01});
        is_initialized = true;
    }
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> observation_noise(const Observation &o) const {
        Eigen::MatrixXd scales = Eigen::MatrixXd::Ones(4, 1);
        double score = o.detection_score, error = o.reprojection_error;
        bool sv = std::isfinite(score) && score >= 0 && score <= 1,
             ev = std::isfinite(error) && error >= 0;
        if (!adaptive_noise || !(sv || ev))
            return {R, scales};
        double quality = (1 + (sv ? (1 - score) * (1 - score) : 0)) *
                         (1 + (ev ? std::pow(std::min(error, 4.) / 2, 2) : 0));
        scales.setConstant(std::min(quality, 3.));
        double grazing = std::pow(std::sin(o.Z_obs(3) + o.Z_obs(0)), 2);
        scales(2) *= 1 + .5 * grazing;
        scales(3) *= 1 + grazing;
        Eigen::MatrixXd D = scales.col(0).array().sqrt().matrix().asDiagonal();
        return {D * R * D, scales};
    }
    Innovation innovation(const Eigen::MatrixXd &z, int id,
                          const std::optional<Eigen::MatrixXd> &noise = std::nullopt) const {
        Eigen::MatrixXd pred = h(X, id), res = residual(z, pred), H = get_jacobian(X, id),
               S = H * P * H.transpose() + (noise ? *noise : R);
        double nis = (res.transpose() * solve(S, res))(0, 0);
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
                auto [noise, scales] = observation_noise(o);
                d.scales = scales;
                for (int id = 0; id < 4; ++id) {
                    if (o.armor_id && *o.armor_id != id)
                        continue;
                    auto in = innovation(o.Z_obs, id, noise);
                    Eigen::MatrixXd base = in.H * P * in.H.transpose();
                    double cost = in.nis + logabsdet(base + noise) - logabsdet(base + R);
                    if (!std::isfinite(d.nis) || in.nis < d.nis) {
                        d.nis = in.nis;
                        d.best_candidate_id = id;
                        d.distance_residual = in.residual(2);
                    }
                    if (in.nis <= nis_gate && std::abs(in.residual(2)) <= max_distance_error)
                        options.push_back({index, id, in.nis, in.residual, in.H, in.prediction,
                                           o.Z_obs, noise, cost});
                }
                d.reason = options.empty() ? "innovation_gate" : "association_conflict";
            }
            candidates.push_back(options);
            diagnostics.push_back(d);
        }
        std::vector<Match> best, selected;
        double best_score = std::numeric_limits<double>::infinity();
        std::function<void(size_t, unsigned, double)> search = [&](size_t index, unsigned used,
                                                                   double score) {
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
                    search(index + 1, used | (1u << c.armor_id), score + c.association_cost);
                    selected.pop_back();
                }
            }
            search(index + 1, used, score);
        };
        search(0, 0, 0);
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
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, 11), res = Eigen::MatrixXd::Zero(n, 1), noise = Eigen::MatrixXd::Zero(n, n);
        for (size_t k = 0; k < matches.size(); ++k) {
            const auto offset = static_cast<Eigen::Index>(4 * k);
            res.block(offset, 0, 4, 1) = matches[k].residual;
            H.block(offset, 0, 4, 11) = matches[k].H;
            noise.block(offset, offset, 4, 4) = matches[k].R;
        }
        Eigen::MatrixXd S = H * P * H.transpose() + noise, K = solve(S, H * P).transpose();
        X = X + K * res;
        X(6) = wrap_to_pi(X(6));
        X(8) = std::clamp(X(8), .20, .30);
        X(9) = std::clamp(X(8) + X(9), .20, .30) - X(8);
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(11, 11) - K * H;
        P = A * P * A.transpose() + K * noise * K.transpose();
        P = ((P + P.transpose()) * .5).eval();
        return result;
    }
    std::optional<int> find_best_armor_id(const Eigen::MatrixXd &z) const {
        auto result = associate({Observation{z}});
        if (result.first.empty())
            return std::nullopt;
        return result.first[0].armor_id;
    }
    std::optional<int> update(const Eigen::MatrixXd &z, std::optional<int> id = std::nullopt) {
        auto result = update_multi({Observation{z, id}});
        if (result.first.empty())
            return std::nullopt;
        return result.first[0].armor_id;
    }
};
} // namespace predictor
