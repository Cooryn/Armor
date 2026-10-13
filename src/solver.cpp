#include "solver.hpp"

#include <cmath>
#include <algorithm>

// 装甲板尺寸配置
const float armor_width_m = 0.135f;
const float armor_height_m = 0.056f;

// PnP求解装甲板位姿
bool solver::solve(::armor &armor, double yaw_hint)
{
    // 角点顺序：左上、左下、右下、右上
    static const std::vector<cv::Point3f> object_points = {
        {-armor_width_m / 2, -armor_height_m / 2, 0},
        {-armor_width_m / 2, armor_height_m / 2, 0},
        {armor_width_m / 2, armor_height_m / 2, 0},
        {armor_width_m / 2, -armor_height_m / 2, 0}};

    // 将检测出的四个角点复制到image_points中
    std::vector<cv::Point2f> image_points(armor.vertices, armor.vertices + 4);

    const double light_length = (cv::norm(image_points[0] - image_points[1]) + cv::norm(image_points[2] - image_points[3])) / 2; // 计算灯条平均长度

    const double gate = std::max(2.0, light_length * .05); // 计算PnP重投影误差阈值

    // 初始化PnP候选解列表
    struct Pose
    {
        cv::Mat rvec, tvec;
        double error, yaw;
    };
    std::vector<Pose> candidates;

    // 候选解检查函数
    auto add = [&](const cv::Mat &rvecs, const cv::Mat &tvecs)
    {
        // 检查深度是否为正
        cv::Mat rotation;
        cv::Rodrigues(rvecs, rotation);
        for (const auto &point : object_points)
        {
            double depth = rotation.at<double>(2, 0) * point.x + rotation.at<double>(2, 1) * point.y + tvecs.at<double>(2);
            if (depth <= 0)
                return;
        }

        // 计算RMS重投影误差
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points, rvecs, tvecs, camera_matrix, distort_coeffs, projected);
        double error = 0;
        for (int i = 0; i < static_cast<int>(projected.size()); ++i)
        {
            const auto residual = image_points[i] - projected[i];
            error += residual.dot(residual);
        }
        error = std::sqrt(error / projected.size());
        if (error > gate)
            return;

        // 检查候选解是否重复
        const double yaw = std::atan2(rotation.at<double>(2, 0), rotation.at<double>(2, 2)) * 180 / CV_PI;
        for (auto &p : candidates)
        {
            if (cv::norm(rvecs - p.rvec) < 1e-4 && cv::norm(tvecs - p.tvec) < 1e-5)
            {
                if (error < p.error)
                    p = {rvecs.clone(), tvecs.clone(), error, yaw}; // 重复则保留误差更小的解
                return;
            }
        }
        candidates.push_back({rvecs.clone(), tvecs.clone(), error, yaw}); // 不重复则直接保存
    };

    // 使用PnP获取平面姿态候选解，并检查候选解的有效性
    std::vector<cv::Mat> rvecs, tvecs;
    cv::solvePnPGeneric(object_points, image_points, camera_matrix, distort_coeffs,
                        rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
    for (int i = 0; i < static_cast<int>(rvecs.size()); ++i)
    {
        add(rvecs[i], tvecs[i]);
    }

    if (candidates.empty())
        return false;

    // 误差差距超过 0.1 像素时，选误差更小的；误差接近时，可以选 yaw 更连续、误差稍大的解

    // 找重投影误差最小的候选解
    int best = 0;
    for (int i = 1; i < static_cast<int>(candidates.size()); ++i)
        if (candidates[i].error < candidates[best].error)
            best = i;
    int selected = best;

    // 选择yaw连续性更好的解
    if (std::isfinite(yaw_hint))
    {
        double difference = std::abs(std::remainder(candidates[best].yaw - yaw_hint, 360.0));
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
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
    return true;
}

// 前后帧装甲板匹配
std::vector<int> solver::temporal_matches(const std::vector<::armor> &armors, double timestamp_s) const
{
    std::vector<int> matches(armors.size(), -1);          // 初始化全部为未匹配
    const double dt = timestamp_s - previous_timestamp_s; // 计算当前帧和上一帧的时间差
    if (dt <= 0 || dt > .1 || previous.empty())
        return matches; // 判断上一帧信息是否可用

    for (int i = 0; i < static_cast<int>(armors.size()); ++i)
    {
        const double width = cv::norm(armors[i].vertices[3] - armors[i].vertices[0]); // 装甲板宽度
        double best = std::numeric_limits<double>::infinity();
        for (int j = 0; j < static_cast<int>(previous.size()); ++j) // 遍历上一帧装甲板
        {
            const auto &track = previous[j];
            const auto expected_center = track.center + track.velocity * static_cast<float>(dt); // 预测当前装甲板中心位置
            const double cost = (width > .5 * track.width && width < 2 * track.width) ? cv::norm(armors[i].center - expected_center) / std::max(10., track.width) : 1e9; // 计算匹配代价
            if (cost < best)
            {
                best = cost;
                matches[i] = static_cast<int>(j); // 更新最佳匹配
            }
        }
        if (best > .6)  
            matches[i] = -1;
    }
    return matches;
}

// 根据上一帧装甲板信息计算yaw_hints
std::vector<double> solver::yaw_hints(const std::vector<::armor> &armors, double timestamp_s) const
{
    const auto matches = temporal_matches(armors, timestamp_s);
    std::vector<double> hints(armors.size(), std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < static_cast<int>(armors.size()); ++i)
        if (matches[i] >= 0)
        {
            const auto &previous_track = previous[matches[i]];
            hints[i] = previous_track.yaw + previous_track.yaw_rate * (timestamp_s - previous_timestamp_s);
        }
    return hints;
}

// 整理新的Track
void solver::finish_frame(const std::vector<::armor> &armors, double timestamp_s)
{
    const auto matches = temporal_matches(armors, timestamp_s); // 解出上一帧对应装甲板
    std::vector<SolverTrack> next;

    // 对当前帧所有装甲板进行Track更新
    for (int i = 0; i < static_cast<int>(armors.size()); ++i)
    {
        const auto &armor = armors[i];
        SolverTrack track{armor.center, {0, 0}, cv::norm(armor.vertices[3] - armor.vertices[0]), armor.yaw, 0}; // 初始化Track

        // 如果匹配到上一帧的Track，则更新速度和角速度
        if (matches[i] >= 0)
        {
            const auto &previous_track = previous[matches[i]];
            const double dt = timestamp_s - previous_timestamp_s;
            track.velocity = (armor.center - previous_track.center) * static_cast<float>(1 / dt);
            const double delta_yaw = std::remainder(armor.yaw - previous_track.yaw, 360.0);
            if (std::abs(delta_yaw) < 35)
                track.yaw_rate = .5 * previous_track.yaw_rate + .5 * delta_yaw / dt;
        }
        next.push_back(track);
    }
    // 保存Track信息
    previous = std::move(next);
    previous_timestamp_s = timestamp_s;
}

void solver::use_video_profile(int profile)
{
    if (profile == 2)
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0, 732.488057, 0, 1714.616882, 546.930868, 0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -.119922, -.078593, .007511, -.028028, 0);
    }
    else
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                         0, 1288.1400736562441, 483.6163720308021, 0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -.47562935060124745, .21831745829617311,
                          .0004957613589406044, -.00034617769548693592, 0);
    }
}

// 完整PnP流程
SolvedFrame solver::solve_frame(std::vector<::armor> armors, double timestamp_ms)
{
    const double timestamp_s = timestamp_ms / 1000;
    const auto hints = yaw_hints(armors, timestamp_s);
    SolvedFrame result;
    for (int i = 0; i < static_cast<int>(armors.size()); ++i)
    {
        auto &armor = armors[i];
        if (!solve(armor, hints[i]))
            continue;
        const Eigen::Vector3d position(armor.tvec.at<double>(0), armor.tvec.at<double>(1), armor.tvec.at<double>(2));
        const Eigen::Vector4d z(std::atan2(position(0), position(2)), std::atan2(position(1), std::hypot(position(0), position(2))),position.norm(), std::remainder(armor.yaw * CV_PI / 180, 2 * CV_PI));
        result.observations.push_back(z);
        result.armors.push_back(std::move(armor));
    }
    finish_frame(result.armors, timestamp_s);
    return result;
}