#include "solver.hpp"
#include <cmath>
#include <algorithm>

// 构造函数
Solver::Solver(const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs)
{
    this->camera_matrix = camera_matrix;
    this->distort_coeffs = distort_coeffs;

    const double LIGHT_HEIGHT = 0.056;
    const double LIGHT_WIDTH = 0.135;
    float half_x = static_cast<float>(LIGHT_WIDTH / 2.0);
    float half_y = static_cast<float>(LIGHT_HEIGHT / 2.0);

    object_points.clear();
    object_points.emplace_back(-half_x, -half_y, 0.f); // 左上
    object_points.emplace_back(-half_x, half_y, 0.f);  // 左下
    object_points.emplace_back(half_x, half_y, 0.f);   // 右下
    object_points.emplace_back(half_x, -half_y, 0.f);  // 右上
}

// PnP求解装甲板位姿
bool Solver::solve(Armor &armor, double yaw_hint)
{
    // 复制检测出的四个角点到image_points中
    std::vector<cv::Point2f> image_points(armor.vertices, armor.vertices + 4);

    // 检查角点输入
    for (const auto &p : image_points)
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            return false;
    if (!cv::isContourConvex(image_points) || std::abs(cv::contourArea(image_points)) < 4)
        return false;

    const double light_length = (cv::norm(image_points[0] - image_points[1]) + cv::norm(image_points[2] - image_points[3])) / 2; // 计算灯条平均长度

    const double gate = std::max(2.0, light_length * .05); // 计算PnP重投影误差阈值
    struct Pose
    {
        cv::Mat rvec, tvec;
        double error, yaw;
    };
    std::vector<Pose> candidates; // 初始化PnP候选解列表

    // 候选解检查函数
    auto add = [&](const cv::Mat &rvec, const cv::Mat &tvec)
    {
        // 检查深度是否为正
        cv::Mat rotation;
        cv::Rodrigues(rvec, rotation);
        for (const auto &point : object_points)
        {
            double depth = rotation.at<double>(2, 0) * point.x + rotation.at<double>(2, 1) * point.y + tvec.at<double>(2);
            if (depth <= 0)
                return;
        }

        // RMS重投影误差
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points, rvec, tvec, camera_matrix, distort_coeffs, projected);
        double error = 0;
        for (size_t i = 0; i < projected.size(); ++i)
        {
            const auto residual = image_points[i] - projected[i];
            error += residual.dot(residual);
        }
        error = std::sqrt(error / projected.size());
        if (!std::isfinite(error) || error > gate)
            return;

        // 检查候选解是否重复
        const double yaw = std::atan2(rotation.at<double>(2, 0), rotation.at<double>(2, 2)) * 180 / CV_PI;
        for (auto &p : candidates)
        {
            if (cv::norm(rvec - p.rvec) < 1e-4 && cv::norm(tvec - p.tvec) < 1e-5)
            {
                if (error < p.error)
                    p = {rvec.clone(), tvec.clone(), error, yaw}; // 保留误差更小的解
                return;
            }
        }
        candidates.push_back({rvec.clone(), tvec.clone(), error, yaw});
    };

    // 使用PnP获取平面姿态候选解，并检查候选解的有效性
    std::vector<cv::Mat> rvec, tvec;
    cv::solvePnPGeneric(object_points, image_points, camera_matrix, distort_coeffs,
                        rvec, tvec, false, cv::SOLVEPNP_IPPE);
    for (size_t i = 0; i < rvec.size(); ++i)
    {
        if (!cv::checkRange(rvec[i]) || !cv::checkRange(tvec[i]))
            continue;
        add(rvec[i], tvec[i]);
    }

    armor.pnp_candidate_count = static_cast<int>(candidates.size());
    armor.pnp_used_temporal = false;
    if (candidates.empty())
        return false;

    // 找重投影误差最小的候选解
    size_t best = 0;
    for (size_t i = 1; i < candidates.size(); ++i)
        if (candidates[i].error < candidates[best].error)
            best = i;
    size_t selected = best;

    // 选择yaw连续性更好的解
    if (std::isfinite(yaw_hint))
    {
        double difference = std::abs(std::remainder(candidates[best].yaw - yaw_hint, 360.0));
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const double angle = std::abs(std::remainder(candidates[i].yaw - yaw_hint, 360.0));
            if (candidates[i].error <= candidates[best].error + .10 && angle <= 35 && angle < difference)
            {
                selected = i;
                difference = angle;
            }
        }
    }

    // 保存最终解
    const auto &pose = candidates[selected];
    armor.rvec = pose.rvec.clone();
    armor.tvec = pose.tvec.clone();
    armor.yaw = pose.yaw;
    armor.reprojection_error = pose.error;
    armor.pnp_used_temporal = selected != best;
    return true;
}

// 前后帧装甲板匹配
std::vector<int> Solver::temporalMatches(const std::vector<Armor> &armors, double timestamp) const
{
    std::vector<int> matches(armors.size(), -1); // 初始化全部为未匹配
    const double dt = timestamp - previous_time; // 计算当前帧和上一帧的时间差
    if (!std::isfinite(dt) || dt <= 0 || dt > .1 || previous.empty())
        return matches; // 判断上一帧信息是否可用

    std::vector<std::vector<double>> costs(armors.size(), std::vector<double>(previous.size())); // 建立代价矩阵
    for (size_t i = 0; i < armors.size(); ++i)
        for (size_t j = 0; j < previous.size(); ++j)
        {
            const auto &p = previous[j];
            const double width = cv::norm(armors[i].vertices[3] - armors[i].vertices[0]);
            const auto expected_center = p.center + p.velocity * static_cast<float>(dt);
            costs[i][j] = (width > .5 * p.width && width < 2 * p.width) ? cv::norm(armors[i].center - expected_center) / std::max(10., p.width) : 1e9; // 代价函数
        }

    // 比较代价
    for (size_t i = 0; i < armors.size(); ++i)
    {
        const size_t j = std::min_element(costs[i].begin(), costs[i].end()) - costs[i].begin();
        const double best = costs[i][j];
        if (best > .6)
            continue;
        matches[i] = static_cast<int>(j);
    }
    return matches;
}

// 根据上一帧装甲板信息计算yaw_hints
std::vector<double> Solver::yawHints(const std::vector<Armor> &armors, double timestamp) const
{
    const auto matches = temporalMatches(armors, timestamp);
    std::vector<double> hints(armors.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < armors.size(); ++i)
        if (matches[i] >= 0)
        {
            const auto &p = previous[matches[i]];
            hints[i] = p.yaw + p.yaw_rate * (timestamp - previous_time);
        }
    return hints;
}

// 整理新的Track
void Solver::finishFrame(const std::vector<Armor> &armors, double timestamp)
{
    const auto matches = temporalMatches(armors, timestamp); // 解出上一帧对应装甲板
    std::vector<Track> next;

    // 对当前帧装甲板进行Track更新
    for (size_t i = 0; i < armors.size(); ++i)
    {
        const auto &a = armors[i];
        Track track{a.center, {0, 0}, cv::norm(a.vertices[3] - a.vertices[0]), a.yaw, 0}; // 初始化Track

        // 如果匹配到上一帧的Track，则更新速度和角速度
        if (matches[i] >= 0)
        {
            const auto &p = previous[matches[i]];
            const double dt = timestamp - previous_time;
            track.velocity = (a.center - p.center) * static_cast<float>(1 / dt);
            const double delta_yaw = std::remainder(a.yaw - p.yaw, 360.0);
            if (std::abs(delta_yaw) < 35)
                track.yaw_rate = .5 * p.yaw_rate + .5 * delta_yaw / dt;
        }
        next.push_back(track);
    }
    // 保存Track信息
    previous = std::move(next);
    previous_time = timestamp;
}
