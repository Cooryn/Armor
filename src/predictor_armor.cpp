#include "predictor_armor.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

// 角度归一化
double armor_wrap_to_pi(double a)
{
    double r = std::fmod(a + armor_pi, 2 * armor_pi);
    if (r < 0)
        r += 2 * armor_pi;
    return r - armor_pi;
}

// 计算两个观测向量的差
Eigen::Vector4d armor_angular_residual(const Eigen::Vector4d &a, const Eigen::Vector4d &b)
{
    Eigen::Vector4d d = a - b;
    for (int i : {0, 1, 3})
        d(i) = armor_wrap_to_pi(d(i));
    return d;
}

// 计算状态转移矩阵和过程噪声协方差矩阵
void ArmorEKF::transition(double dt, Eigen::Matrix<double, 11, 11> &f, Eigen::Matrix<double, 11, 11> &q) const
{
    f << 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0,  1, dt,0, 0, 0, 0, 0, 0, 0,
         0, 0,  0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0,  0, 0, 1, dt,0, 0, 0, 0, 0,
         0, 0,  0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0,  0, 0, 0, 0, 1, dt,0, 0, 0,
         0, 0,  0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0,  0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0,  0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 1;

    const double p00 = dt * dt * dt / 3 * q_pos;
    const double p01 = dt * dt / 2 * q_pos;
    const double p11 = dt * q_pos;
    const double y00 = dt * dt * dt / 3 * q_yaw;
    const double y01 = dt * dt / 2 * q_yaw;
    const double y11 = dt * q_yaw;
    q << p00, p01, 0,   0,   0,   0,   0,   0,   0,        0,         0,
         p01, p11, 0,   0,   0,   0,   0,   0,   0,        0,         0,
         0,   0,   p00, p01, 0,   0,   0,   0,   0,        0,         0,
         0,   0,   p01, p11, 0,   0,   0,   0,   0,        0,         0,
         0,   0,   0,   0,   p00, p01, 0,   0,   0,        0,         0,
         0,   0,   0,   0,   p01, p11, 0,   0,   0,        0,         0,
         0,   0,   0,   0,   0,   0,   y00, y01, 0,        0,         0,
         0,   0,   0,   0,   0,   0,   y01, y11, 0,        0,         0,
         0,   0,   0,   0,   0,   0,   0,   0,   q_r * dt, 0,         0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,        q_dl * dt, 0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,        0,         q_dh * dt;
}

void ArmorEKF::predict(double dt)
{
    Eigen::Matrix<double, 11, 11> q;
    transition(dt, F, q);
    X = F * X;
    X(6) = armor_wrap_to_pi(X(6));
    P = F * P * F.transpose() + q; // 更新状态估计的协方差
}

// 预测未来状态
Forecast ArmorEKF::forecast(double horizon_s) const
{
    Eigen::Matrix<double, 11, 11> f, q;
    transition(horizon_s, f, q);
    Eigen::Matrix<double, 11, 1> s = f * X;
    s(6) = armor_wrap_to_pi(s(6));
    Eigen::Matrix4d poses;
    for (int id = 0; id < 4; ++id)
        poses.row(id) = armor_plate_pose(s, id).transpose();
    return {s, f * P * f.transpose() + q, poses};
}

// 计算每块装甲板的位姿
Eigen::Vector4d armor_plate_pose(const Eigen::Matrix<double, 11, 1> &s, int id)
{
    double yaw = s(6) + id * armor_pi / 2;
    double r = s(8) + (id % 2 ? s(9) : 0);
    double x = s(0) + r * std::sin(yaw);
    double y = s(2) + (id % 2 ? s(10) : 0);
    double z = s(4) - r * std::cos(yaw);
    return {x, y, z, armor_wrap_to_pi(yaw)};
}

// 计算观测值
Eigen::Vector4d armor_h(const Eigen::Matrix<double, 11, 1> &s, int id)
{
    const auto pose = armor_plate_pose(s, id);
    const double x = pose(0), y = pose(1), z = pose(2);
    return {armor_wrap_to_pi(std::atan2(x, z)),
            armor_wrap_to_pi(std::atan2(y, std::sqrt(x * x + z * z))),
            std::sqrt(x * x + y * y + z * z), pose(3)};
}

// 计算雅可比矩阵
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

// 首帧初始化
void ArmorEKF::initialize(const Eigen::Vector4d &z)
{
    double x = z(2) * std::cos(z(1)) * std::sin(z(0)), y = z(2) * std::sin(z(1)),
           zc = z(2) * std::cos(z(1)) * std::cos(z(0));
    X << x - .26 * std::sin(z(3)), 0, y, 0, zc + .26 * std::cos(z(3)), 0, z(3), 0, .26, 0, 0;
    P = (Eigen::Matrix<double, 11, 1>() << .1, 1, .1, 1, .1, 1, .05, 100, .01, .01, .01).finished().asDiagonal();
    is_initialized = true;
}


std::vector<Match> ArmorEKF::associate(const std::vector<Eigen::Vector4d> &observations) const
{
    std::vector<Match> candidates, matches;
    for (size_t index = 0; index < observations.size(); ++index)
    {
        const auto &z = observations[index];
        const Eigen::Vector3d position(z(2) * std::cos(z(1)) * std::sin(z(0)),
                                       z(2) * std::sin(z(1)),
                                       z(2) * std::cos(z(1)) * std::cos(z(0)));
        for (int id = 0; id < 4; ++id)
        {
            const Eigen::Vector4d plate = armor_plate_pose(X, id);
            const double distance = (position - plate.head<3>()).norm();
            if (distance <= max_distance_error &&
                std::abs(armor_wrap_to_pi(z(3) - plate(3))) <= max_yaw_error)
                candidates.push_back({index, id, distance, armor_angular_residual(z, armor_h(X, id)), z});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b)
    {
        if (a.distance != b.distance)
            return a.distance < b.distance;
        if (a.index != b.index)
            return a.index < b.index;
        return a.armor_id < b.armor_id;
    });
    for (const auto &candidate : candidates)
    {
        bool conflict = false;
        for (const auto &match : matches)
            if (candidate.index == match.index || candidate.armor_id == match.armor_id ||
                std::abs(armor_wrap_to_pi(candidate.Z_obs(3) - match.Z_obs(3) -
                    (candidate.armor_id - match.armor_id) * armor_pi / 2)) > pair_yaw_tolerance)
            {
                conflict = true;
                break;
            }
        if (conflict)
            continue;
        matches.push_back(candidate);
        if (matches.size() == 4)
            break;
    }
    return matches;
}

std::vector<Match> ArmorEKF::update_multi(const std::vector<Eigen::Vector4d> &obs)
{
    auto matches = associate(obs);
    if (matches.empty())
        return matches;
    int n = static_cast<int>(matches.size()) * 4;
    Eigen::Matrix<double, Eigen::Dynamic, 11, 0, 16, 11> H(n, 11);
    Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 16, 1> res(n);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16> noise(n, n);
    noise.setZero();
    for (size_t k = 0; k < matches.size(); ++k)
    {
        const auto offset = static_cast<Eigen::Index>(4 * k);
        res.segment<4>(offset) = matches[k].residual;
        H.middleRows<4>(offset) = armor_get_jacobian(X, matches[k].armor_id);
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
    return matches;
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
    std::vector<Eigen::Vector4d> all;
    for (const auto &observation : observations)
    {
        all.push_back(observation.measurement);
        if (!selected || observation.measurement(2) < selected->measurement(2))
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
        result.status = selected ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);
        const Eigen::Matrix<double, 11, 1> prior = X;
        const auto matches = update_multi(all);
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
