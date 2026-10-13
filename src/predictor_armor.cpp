#include "predictor_armor.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

const double predictor_armor::pi = 3.14159265358979323846;

// 角度归一化
double predictor_armor::wrap_to_pi(double angle)
{
    double wrapped_angle = std::fmod(angle + predictor_armor::pi, 2 * predictor_armor::pi);
    if (wrapped_angle < 0)
        wrapped_angle += 2 * predictor_armor::pi;
    return wrapped_angle - predictor_armor::pi;
}

// 计算两个观测向量的差值
Eigen::Vector4d predictor_armor::residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted)
{
    Eigen::Vector4d residual = observed - predicted;
    for (int i : {0, 1, 3})
        residual(i) = predictor_armor::wrap_to_pi(residual(i));
    return residual;
}

// 计算dt时间差下的状态转移矩阵F和过程噪声协方差矩阵Q
void predictor_armor::transition(double dt, Eigen::Matrix<double, 11, 11> &F, Eigen::Matrix<double, 11, 11> &Q) const
{
    F.setIdentity();
    Q.setZero();
    const double p00 = dt * dt * dt / 3 * q_pos;
    const double p01 = dt * dt / 2 * q_pos;
    const double p11 = dt * q_pos;
    const double y00 = dt * dt * dt / 3 * q_yaw;
    const double y01 = dt * dt / 2 * q_yaw;
    const double y11 = dt * q_yaw;
    for (int i : {0, 2, 4, 6})
    {
        F(i, i + 1) = dt;
        Q(i, i) = i == 6 ? y00 : p00;
        Q(i, i + 1) = Q(i + 1, i) = i == 6 ? y01 : p01;
        Q(i + 1, i + 1) = i == 6 ? y11 : p11;
    }
    Q(8, 8) = q_r * dt;
    Q(9, 9) = q_dl * dt;
    Q(10, 10) = q_dh * dt;
}

// 预测
void predictor_armor::predict(double dt)
{
    Eigen::Matrix<double, 11, 11> F, Q;
    transition(dt, F, Q);
    X = F * X;
    X(6) = predictor_armor::wrap_to_pi(X(6));
    P = F * P * F.transpose() + Q; // 更新状态估计的协方差
}

// 计算装甲板的位姿
Eigen::Vector4d predictor_armor::plate_pose(const Eigen::Matrix<double, 11, 1> &X, int armor_id)
{
    double yaw = X(6) + armor_id * predictor_armor::pi / 2;
    double r = X(8) + (armor_id % 2 ? X(9) : 0);
    double x = X(0) + r * std::sin(yaw);
    double y = X(2) + (armor_id % 2 ? X(10) : 0);
    double z = X(4) - r * std::cos(yaw);
    return {x, y, z, predictor_armor::wrap_to_pi(yaw)};
}

// 计算估计观测值
Eigen::Vector4d predictor_armor::h(const Eigen::Matrix<double, 11, 1> &X, int armor_id)
{
    const auto pose = predictor_armor::plate_pose(X, armor_id);
    const double x = pose(0), y = pose(1), z = pose(2);
    const double horizontal = std::hypot(x, z);
    return {predictor_armor::wrap_to_pi(std::atan2(x, z)),
            predictor_armor::wrap_to_pi(std::atan2(y, horizontal)),
            std::hypot(horizontal, y), pose(3)};
}

// 计算雅可比矩阵
Eigen::Matrix<double, 4, 11> predictor_armor::jacobian(const Eigen::Matrix<double, 11, 1> &X, int armor_id)
{
    Eigen::Matrix<double, 4, 11> H = Eigen::Matrix<double, 4, 11>::Zero();
    const Eigen::Vector4d predicted_observation = predictor_armor::h(X, armor_id);
    for (int i : {0, 2, 4, 6, 8, 9, 10})
    {
        Eigen::Matrix<double, 11, 1> perturbed_state = X;
        perturbed_state(i) += 1e-5;
        const Eigen::Vector4d residual = predictor_armor::residual(predictor_armor::h(perturbed_state, armor_id), predicted_observation);
        H.col(i) = residual / 1e-5;
    }
    return H;
}

// 首帧初始化
void predictor_armor::initialize(const Eigen::Vector4d &Z)
{
    const double horizontal = Z(2) * std::cos(Z(1));
    const double x = horizontal * std::sin(Z(0)), y = Z(2) * std::sin(Z(1));
    const double z = horizontal * std::cos(Z(0));
    const double yaw = predictor_armor::wrap_to_pi(Z(3));
    X << x - .26 * std::sin(yaw), 0, y, 0, z + .26 * std::cos(yaw), 0, yaw, 0, .26, 0, 0;
    P = (Eigen::Matrix<double, 11, 1>() << .1, 1, .1, 1, .1, 1, .05, 100, .01, .01, .01).finished().asDiagonal();
    is_initialized = true;
}

// 联合多板match
std::vector<Match> predictor_armor::associate(const std::vector<Eigen::Vector4d> &observations) const
{
    std::vector<Match> candidates, matches;

    // 遍历所有观测值
    for (int i = 0; i < static_cast<int>(observations.size()); ++i)
    {
        const auto &Z = observations[i];
        const Eigen::Vector3d position(Z(2) * std::cos(Z(1)) * std::sin(Z(0)), Z(2) * std::sin(Z(1)), Z(2) * std::cos(Z(1)) * std::cos(Z(0))); // 计算观测值的三维位置

        // 尝试匹配每块装甲板
        for (int armor_id = 0; armor_id < 4; ++armor_id)
        {
            const Eigen::Vector4d plate = predictor_armor::plate_pose(X, armor_id); // [x,y,z,yaw]
            if (std::hypot(plate(0), plate(2)) <= 1e-6)
                continue;
            const double distance = (position - plate.head<3>()).norm();
            if (distance <= max_distance_error && std::abs(predictor_armor::wrap_to_pi(Z(3) - plate(3))) <= max_yaw_error)
                candidates.push_back({i, armor_id, distance, predictor_armor::residual(Z, predictor_armor::h(X, armor_id))});
        }
    }

    // 对候选匹配进行排序，优先选择距离较近的匹配
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b)
              {
        if (a.distance != b.distance)
            return a.distance < b.distance;
        if (a.index != b.index)
            return a.index < b.index;
        return a.armor_id < b.armor_id; });

    // 去除冲突并保存到matches中
    for (const auto &candidate : candidates)
    {
        bool conflict = false;
        for (const auto &match : matches)
        {
            if (candidate.index == match.index || candidate.armor_id == match.armor_id)
            {
                conflict = true;
                break;
            }
        }
        if (conflict)
            continue;
        matches.push_back(candidate);
        if (matches.size() == 4)
            break;
    }
    return matches;
}

// 联合多板更新EKF
std::vector<Match> predictor_armor::update(const std::vector<Eigen::Vector4d> &observations)
{
    auto matches = associate(observations);
    if (matches.empty())
        return matches;
    int n = static_cast<int>(matches.size()) * 4;
    Eigen::Matrix<double, Eigen::Dynamic, 11, 0, 16, 11> H(n, 11);            // 创建联合雅可比矩阵
    Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 16, 1> residual(n);           // 创建联合残差列向量
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16> R(n, n); // 创建联合观测噪声协方差矩阵
    R.setZero();

    for (int k = 0; k < static_cast<int>(matches.size()); ++k)
    {
        const auto offset = static_cast<Eigen::Index>(4 * k);
        residual.segment<4>(offset) = matches[k].residual; // 将当前匹配的四维残差填入联合残差向量
        H.middleRows<4>(offset) = predictor_armor::jacobian(X, matches[k].armor_id);
        R.block<4, 4>(offset, offset) = this->R;
    }

    // 开始联合更新
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16> S = H * P * H.transpose() + R;  // 计算联合残差协方差
    const auto decomposition = S.ldlt();                                                                   // 使用LDLT分解
    const Eigen::Matrix<double, 11, Eigen::Dynamic, 0, 11, 16> K = decomposition.solve(H * P).transpose(); // 计算卡尔曼增益
    X = X + K * residual;                                                                                  // 修正状态
    X(6) = predictor_armor::wrap_to_pi(X(6));
    X(8) = std::clamp(X(8), .20, .30);
    X(9) = std::clamp(X(8) + X(9), .20, .30) - X(8);
    const Eigen::Matrix<double, 11, 11> A = Eigen::Matrix<double, 11, 11>::Identity() - K * H;
    P = A * P * A.transpose() + K * R * K.transpose(); // 更新协方差
    P = ((P + P.transpose()) * .5).eval();             // 消除不对称误差
    return matches;
}

// 更新帧
PredictionResult predictor_armor::update_frame(double timestamp_ms, const std::vector<Eigen::Vector4d> &observations, double horizon_ms)
{
    const double dt = (timestamp_ms - last_timestamp_ms) / 1000; // 计算时间间隔
    last_timestamp_ms = timestamp_ms;                            // 记录时间戳

    const double missing = std::numeric_limits<double>::quiet_NaN(); // 定义缺失值
    PredictionResult prediction;
    const Eigen::Vector4d *selected_observation = nullptr;

    // 选择距离最近的观测值
    for (const auto &Z : observations)
    {
        if (!selected_observation || Z(2) < (*selected_observation)(2))
            selected_observation = &Z;
    }

    Eigen::Vector4d plate = Eigen::Vector4d::Constant(missing);
    double observed_yaw = missing;

    // 更新
    if (!is_initialized)
    {
        if (selected_observation)
            initialize(*selected_observation);
        prediction.status = selected_observation ? "initialized" : "waiting";
    }
    else
    {
        predict(dt);                                        // 把状态和协方差推进到当前帧时间
        const Eigen::Matrix<double, 11, 1> prior_state = X; // 保存状态
        const auto matches = update(observations);          // 联合更新
        if (!matches.empty())
        {
            // 找到观测距离最小的记录误差
            const auto &match = *std::min_element(matches.begin(), matches.end(), [&observations](const auto &a, const auto &b)
                                                  { return observations[a.index](2) < observations[b.index](2); });
            plate = predictor_armor::plate_pose(prior_state, match.armor_id);
            prediction.errors = -match.residual;
            observed_yaw = observations[match.index](3);
        }
        prediction.status = matches.empty() ? "prediction_only" : "updated";
    }

    // 生成几何结果
    if (is_initialized)
    {
        // 计算未来状态和四板位姿
        auto future = X;
        for (int i : {0, 2, 4, 6})                          // [x, y, z, yaw]
            future(i) += horizon_ms / 1000 * future(i + 1); // 预测未来位置
        future(6) = predictor_armor::wrap_to_pi(future(6));
        prediction.current.center = Eigen::Vector3d(X(0), X(2), X(4));
        prediction.future.center = Eigen::Vector3d(future(0), future(2), future(4));
        for (int i = 0; i < 4; ++i)
        {
            // 保存当前和未来的四板位姿
            prediction.current.plates.push_back(predictor_armor::plate_pose(X, i));
            prediction.future.plates.push_back(predictor_armor::plate_pose(future, i));
        }
        if (prediction.status != "initialized")
        {
            // 填写CSV输出值
            prediction.values = {future(6), plate(0), plate(2), X(6), plate(3), observed_yaw, X(8), X(9), X(10)};
        }
    }
    return prediction;
}
