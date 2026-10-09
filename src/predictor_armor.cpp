#include "predictor_armor.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

// Fixed observation standard deviations: [0.04 rad, 0.04 rad, 0.40 m, 0.24 rad].

double armor_wrap_to_pi(double a)
{
    double r = std::fmod(a + armor_pi, 2 * armor_pi);
    if (r < 0)
        r += 2 * armor_pi;
    return r - armor_pi;
}

Eigen::Vector4d armor_angular_residual(const Eigen::Vector4d &a, const Eigen::Vector4d &b)
{
    Eigen::Vector4d d = a - b;
    for (int i : {0, 1, 3})
        d(i) = armor_wrap_to_pi(d(i));
    return d;
}

std::pair<Eigen::Matrix<double, 11, 11>, Eigen::Matrix<double, 11, 11>> ArmorEKF::transition(double dt) const
{
    Eigen::Matrix<double, 11, 11> f = Eigen::Matrix<double, 11, 11>::Identity(), q = Eigen::Matrix<double, 11, 11>::Zero();
    for (int i : {0, 2, 4, 6})
    {
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

void ArmorEKF::predict(double dt)
{
    auto [f, q] = transition(dt);
    F = f;
    X = F * X;
    X(6) = armor_wrap_to_pi(X(6));
    P = F * P * F.transpose() + q;
}

Forecast ArmorEKF::forecast(double horizon_s) const
{
    auto [f, q] = transition(horizon_s);
    Eigen::Matrix<double, 11, 1> s = f * X;
    s(6) = armor_wrap_to_pi(s(6));
    Eigen::Matrix4d poses;
    for (int id = 0; id < 4; ++id)
        poses.row(id) = armor_plate_pose(s, id).transpose();
    return {s, f * P * f.transpose() + q, poses};
}

Eigen::Vector4d armor_plate_pose(const Eigen::Matrix<double, 11, 1> &s, int id)
{
    double yaw = s(6) + id * armor_pi / 2, r = s(8) + (id % 2 ? s(9) : 0),
           x = s(0) + r * std::sin(yaw), y = s(2) + (id % 2 ? s(10) : 0),
           z = s(4) - r * std::cos(yaw);
    return {x, y, z, armor_wrap_to_pi(yaw)};
}

Eigen::Vector4d armor_h(const Eigen::Matrix<double, 11, 1> &s, int id)
{
    const auto pose = armor_plate_pose(s, id);
    const double x = pose(0), y = pose(1), z = pose(2);
    return {armor_wrap_to_pi(std::atan2(x, z)),
            armor_wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), pose(3)};
}

Eigen::Matrix<double, 4, 11> armor_get_jacobian(const Eigen::Matrix<double, 11, 1> &s, int id)
{
    Eigen::Matrix<double, 4, 11> H = Eigen::Matrix<double, 4, 11>::Zero();
    const Eigen::Vector4d base = armor_h(s, id);
    for (int i : {0, 2, 4, 6, 8, 9, 10})
    {
        Eigen::Matrix<double, 11, 1> t = s;
        t(i) += 1e-5;
        const Eigen::Vector4d d = armor_angular_residual(armor_h(t, id), base);
        H.col(i) = d / 1e-5;
    }
    return H;
}

bool ArmorEKF::valid_observation(const Eigen::Vector4d &z) const
{
    return z(2) > 0 &&
           std::abs(z(1)) < armor_pi / 2 && (base_frame || std::abs(z(0)) < armor_pi / 2);
}

void ArmorEKF::initialize(const Eigen::Vector4d &z)
{
    double x = z(2) * std::cos(z(1)) * std::sin(z(0)), y = z(2) * std::sin(z(1)),
           zc = z(2) * std::cos(z(1)) * std::cos(z(0));
    X << x - .26 * std::sin(z(3)), 0, y, 0, zc + .26 * std::cos(z(3)), 0, z(3), 0, .26, 0, 0;
    P = (Eigen::Matrix<double, 11, 1>() << .1, 1, .1, 1, .1, 1, .05, 100, .01, .01, .01).finished().asDiagonal();
    is_initialized = true;
}

std::pair<std::vector<Match>, std::vector<Diagnostic>> ArmorEKF::associate(const std::vector<Observation> &observations) const
{
    std::vector<std::vector<Match>> candidates;
    std::vector<Diagnostic> diagnostics;
    for (size_t index = 0; index < observations.size(); ++index)
    {
        const auto &o = observations[index];
        Diagnostic d;
        d.observation_index = index;
        std::vector<Match> options;
        if (valid_observation(o.Z_obs))
        {
            for (int id = 0; id < 4; ++id)
            {
                if (o.armor_id != -1 && o.armor_id != id)
                    continue;
                const Eigen::Vector4d pred = armor_h(X, id), res = armor_angular_residual(o.Z_obs, pred);
                const Eigen::Matrix<double, 4, 11> H = armor_get_jacobian(X, id);
                const Eigen::Matrix4d S = H * P * H.transpose() + R;
                const Eigen::LDLT<Eigen::Matrix4d> solver(S);
                const double nis = res.dot(solver.solve(res));
                if (d.best_candidate_id < 0 || nis < d.nis)
                {
                    d.nis = nis;
                    d.best_candidate_id = id;
                    d.distance_residual = res(2);
                }
                if (nis <= nis_gate && std::abs(res(2)) <= max_distance_error)
                    options.push_back({index, id, nis, res, H, o.Z_obs});
            }
            d.reason = options.empty() ? "innovation_gate" : "association_conflict";
        }
        candidates.push_back(options);
        diagnostics.push_back(d);
    }
    std::vector<Match> best, selected;
    double best_score = std::numeric_limits<double>::infinity();
    auto search = [&](auto &&recurse, size_t index, unsigned used, double score) -> void
    {
        if (selected.size() + std::min(size_t(4) - selected.size(), candidates.size() - index) <
            best.size())
            return;
        if (index == candidates.size())
        {
            if (selected.size() > best.size() ||
                (selected.size() == best.size() && score < best_score))
            {
                best = selected;
                best_score = score;
            }
            return;
        }
        for (const auto &c : candidates[index])
        {
            if (used & (1u << c.armor_id))
                continue;
            bool consistent = true;
            for (const auto &o : selected)
                if (std::abs(armor_wrap_to_pi(c.Z_obs(3) - o.Z_obs(3) -
                                              (c.armor_id - o.armor_id) * armor_pi / 2)) >
                    pair_yaw_tolerance)
                {
                    consistent = false;
                    break;
                }
            if (consistent)
            {
                selected.push_back(c);
                recurse(recurse, index + 1, used | (1u << c.armor_id), score + c.nis);
                selected.pop_back();
            }
        }
        recurse(recurse, index + 1, used, score);
    };
    search(search, 0, 0, 0);
    for (const auto &m : best)
    {
        auto &d = diagnostics[m.index];
        d.accepted = true;
        d.armor_id = m.armor_id;
        d.nis = m.nis;
        d.reason = "accepted";
        d.distance_residual = m.residual(2);
    }
    return {best, diagnostics};
}

std::pair<std::vector<Match>, std::vector<Diagnostic>> ArmorEKF::update_multi(const std::vector<Observation> &obs)
{
    auto result = associate(obs);
    auto &matches = result.first;
    if (matches.empty())
        return result;
    int n = static_cast<int>(matches.size()) * 4;
    Eigen::Matrix<double, Eigen::Dynamic, 11, 0, 16, 11> H(n, 11);
    Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 16, 1> res(n);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16> noise(n, n);
    noise.setZero();
    for (size_t k = 0; k < matches.size(); ++k)
    {
        const auto offset = static_cast<Eigen::Index>(4 * k);
        res.segment<4>(offset) = matches[k].residual;
        H.middleRows<4>(offset) = matches[k].H;
        noise.block<4, 4>(offset, offset) = R;
    }
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16> S = H * P * H.transpose() + noise;
    const auto solver = S.ldlt();
    const Eigen::Matrix<double, 11, Eigen::Dynamic, 0, 11, 16> K = solver.solve(H * P).transpose();
    X = X + K * res;
    X(6) = armor_wrap_to_pi(X(6));
    X(8) = std::clamp(X(8), .20, .30);
    X(9) = std::clamp(X(8) + X(9), .20, .30) - X(8);
    const Eigen::Matrix<double, 11, 11> A = Eigen::Matrix<double, 11, 11>::Identity() - K * H;
    P = A * P * A.transpose() + K * noise * K.transpose();
    P = ((P + P.transpose()) * .5).eval();
    return result;
}

VideoPrediction ArmorEKF::update_frame(std::int64_t frame, double timestamp_ms,
                                      const std::vector<VideoObservation> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms_) / 1000;
    last_timestamp_ms_ = timestamp_ms;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    VideoPrediction result;
    result.horizon_ms = horizon_ms;
    const VideoObservation *selected = nullptr;
    std::vector<Observation> all;
    for (const auto &observation : observations)
    {
        all.push_back({observation.measurement});
        if (valid_observation(observation.measurement) &&
            (!selected || observation.measurement(2) < selected->measurement(2)))
            selected = &observation;
    }
    const bool initialized_before = is_initialized;
    Eigen::Vector4d plate = Eigen::Vector4d::Constant(missing);
    double observed_yaw = missing;
    int id = -1;
    std::size_t accepted_count = 0;
    if (!is_initialized)
    {
        if (selected)
            initialize(selected->measurement);
        for (std::size_t i = 0; i < observations.size(); ++i)
        {
            Diagnostic diagnostic;
            diagnostic.observation_index = i;
            diagnostic.accepted = &observations[i] == selected;
            diagnostic.armor_id = diagnostic.accepted ? 0 : -1;
            diagnostic.reason = !valid_observation(observations[i].measurement) ? "invalid" :
                                diagnostic.accepted ? "initialization" : "initialization_unused";
            result.diagnostics.push_back(diagnostic);
        }
        result.status = selected ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        const Eigen::Matrix<double, 11, 1> prior = X;
        const auto [matches, diagnostics] = update_multi(all);
        result.diagnostics = diagnostics;
        accepted_count = matches.size();
        if (!matches.empty())
        {
            const auto &match = *std::min_element(matches.begin(), matches.end(),
                [](const auto &a, const auto &b) { return a.Z_obs(2) < b.Z_obs(2); });
            id = match.armor_id;
            plate = armor_plate_pose(prior, id);
            result.errors = -match.residual;
            observed_yaw = match.Z_obs(3);
        }
        result.status = matches.empty() ? "prediction_only" : "updated";
    }
    result.initialized = is_initialized;
    if (is_initialized)
    {
        result.position_variance = std::max({P(0, 0), P(2, 2), P(4, 4)});
        const auto current = forecast(0), future = forecast(horizon_ms / 1000);
        result.current.center = Eigen::Vector3d(X(0), X(2), X(4));
        result.future.center = Eigen::Vector3d(future.state(0), future.state(2), future.state(4));
        for (int i = 0; i < 4; ++i)
        {
            result.current.plates.push_back(current.plates.row(i).transpose());
            result.future.plates.push_back(future.plates.row(i).transpose());
        }
        if (initialized_before)
        {
            const auto &f = future.state;
            const auto &errors = result.errors;
            result.values = {double(frame), timestamp_ms, horizon_ms, timestamp_ms + horizon_ms,
                             f(0), f(2), f(4), f(6), X(0), X(2), X(4), X(1), X(3), X(5), X(7), plate(0), plate(2),
                             double(id), X(6), plate(3), observed_yaw, errors(0), errors(1), errors(2), errors(3),
                             X(8), X(9), X(10), double(all.size()), double(accepted_count), double(all.size() - accepted_count)};
            result.value_count = 31;
        }
    }
    return result;
}
