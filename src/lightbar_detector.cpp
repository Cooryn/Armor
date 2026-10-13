#include "lightbar_detector.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

// 提取灯条mask
cv::Mat detector::extract_color(const cv::Mat &image, EnemyColor color, int color_th, int gray_th)
{
    // 分离BGR通道
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat color_mask;

    // 颜色差分+灰度阈值过滤
    if (color == ENEMY_RED)
    {
        cv::Mat r_sub_b;
        cv::subtract(channels[2], channels[0], r_sub_b);
        color_mask = (r_sub_b > color_th) & (channels[2] > gray_th);
    }
    else
    {
        cv::Mat b_sub_r;
        cv::subtract(channels[0], channels[2], b_sub_r);
        color_mask = (b_sub_r > color_th) & (channels[0] > gray_th);
    }

    // 将图片二值化，提取出高亮部分
    cv::Mat gray, highlight_mask;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, highlight_mask, 210, 255, cv::THRESH_BINARY);

    // 处理过曝
    cv::Mat shield;
    static const cv::Mat big_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::dilate(color_mask, shield, big_kernel);              // 膨胀颜色区域
    cv::bitwise_and(highlight_mask, shield, highlight_mask); // 高亮区域与膨胀后的颜色区域相交
    cv::bitwise_or(color_mask, highlight_mask, color_mask);  // 合并颜色区域和相交后的高亮区域

    // 闭运算填补空隙
    static const cv::Mat small_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, small_kernel);

    return color_mask;
}

// 筛选有效灯条的旋转矩形
std::vector<cv::RotatedRect> detector::get_valid_light_rects(
    const std::vector<std::vector<cv::Point>> &light_bars, std::vector<float> &quality,
    float min_angle,
    double minAspectRatio, double minArea)
{
    std::vector<cv::RotatedRect> rects;

    // 清空拟合质量
    quality.clear();

    for (const auto &c : light_bars)
    {
        // 硬性条件筛选轮廓
        if (c.size() < 3)
            continue;
        const double area = cv::contourArea(c);
        if (area < minArea)
            continue;

        // 初始化最小外接旋转矩形
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;

        // 几何条件筛选旋转矩形
        if (w < 1.f || h < 1.f || std::max(w, h) / std::min(w, h) < minAspectRatio || area / (w * h) < .30)
            continue;
        float angle = std::abs(rect.angle);
        float longEdgeAngle = (w >= h) ? angle : (90.0f - angle);
        if (longEdgeAngle < min_angle)
            continue;

        // 初始拟合质量
        float fit_quality = .75f;

        // 宽轮廓保留旋转矩形，对细长轮廓做fitline
        if (std::max(w, h) >= 2.f * std::min(w, h))
        {
            const std::vector<cv::Point2f> boundary(c.begin(), c.end()); // 把轮廓点转换成Point2f
            cv::Vec4f line;                                              // 初始化拟合直线
            cv::fitLine(boundary, line, cv::DIST_HUBER, 0, .01, .01);
            cv::Point2f axis(line[0], line[1]); // 灯条长轴方向向量
            if (axis.y < 0)
                axis *= -1.f; // 统一方向向下
            const cv::Point2f origin(line[2], line[3]);

            // 角度比较
            float fitted_angle = std::atan2(axis.y, axis.x) * static_cast<float>(180.0 / CV_PI);
            const float old_angle = rect.angle + (w < h ? 90.f : 0.f);
            const float axis_change = std::abs(std::remainder(fitted_angle - old_angle, 180.f));

            // 角度变化过大直接舍弃
            if (axis_change > 10.f)
                continue;

            fitted_angle = old_angle + .5f * std::remainder(fitted_angle - old_angle, 180.f); // 取平均值

            // 重新构造单位方向向量
            const float radians = fitted_angle * static_cast<float>(CV_PI / 180.0);
            axis = cv::Point2f(std::cos(radians), std::sin(radians));
            if (axis.y < 0)
                axis *= -1.f;
            const cv::Point2f normal(axis.y, -axis.x); // 构造单位法向量

            // 计算灯条实际长度
            float along_min = std::numeric_limits<float>::infinity();
            float along_max = -std::numeric_limits<float>::infinity();
            float across_min = std::numeric_limits<float>::infinity();
            float across_max = -std::numeric_limits<float>::infinity();
            for (const auto &point : boundary)
            {
                const float projection = (point - origin).dot(axis); // 计算轮廓点到origin的长轴投影
                along_min = std::min(along_min, projection);
                along_max = std::max(along_max, projection);
                const float across = (point - origin).dot(normal); // 计算轮廓点到origin的短轴投影
                across_min = std::min(across_min, across);
                across_max = std::max(across_max, across);
            }
            const float length = along_max - along_min;                               // 灯条长度
            const float width = across_max - across_min;                              // 灯条宽度
            const cv::Point2f center = origin + axis * ((along_min + along_max) / 2); // 灯条中心
            rect = cv::RotatedRect(center, cv::Size2f(length, width), fitted_angle);  // 重构旋转矩形
            fit_quality = 1.f;
        }
        const float fill = static_cast<float>(area) / rect.size.area();                // 计算填充率
        quality.push_back(std::sqrt(fit_quality * std::clamp(fill / .75f, .5f, 1.f))); // 计算灯条质量
        rects.push_back(rect);
    }
    return rects;
}

// 重定义灯条集合信息
struct LightGeometry
{
    float length, width;
    cv::Point2f axis;
    cv::Point2f top, bottom;
};

// 定义候选装甲板
struct Candidate
{
    int left, right;
    double score;
};

// 灯条匹配装甲板
std::vector<::armor> detector::match_armors(const std::vector<cv::RotatedRect> &light_bars,
                                            const std::vector<float> &light_quality,
                                            float max_angle_diff,
                                            float max_length_ratio,
                                            float min_aspect_ratio,
                                            float max_y_diff_ratio,
                                            float max_aspect_ratio,
                                            float min_detection_score)
{
    // 按照灯条中心位置从左往右、从上到下排序，将顺序存入order
    std::vector<int> order(light_bars.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b)
              {
        if (light_bars[a].center.x != light_bars[b].center.x)
            return light_bars[a].center.x < light_bars[b].center.x;
        return light_bars[a].center.y < light_bars[b].center.y; });

    // 初始化LightGeometry对象
    std::vector<LightGeometry> lights;
    for (const auto index : order)
    {
        const auto &bar = light_bars[index];
        const float angle = (bar.angle + (bar.size.width < bar.size.height ? 90.f : 0.f)) * static_cast<float>(CV_PI / 180.0);
        cv::Point2f axis(std::cos(angle), std::sin(angle));
        if (axis.y < 0)
            axis *= -1.f;
        const float length = std::max(bar.size.width, bar.size.height);
        lights.push_back({length, std::min(bar.size.width, bar.size.height), axis,
                          bar.center - axis * (length / 2), bar.center + axis * (length / 2)});
    }

    std::vector<Candidate> candidates; // 初始化候选装甲板集合

    // 只配对相邻的灯条
    for (int i = 0; i + 1 < static_cast<int>(order.size()); ++i)
    {
        const int j = i + 1;
        const auto &l = lights[i];
        const auto &r = lights[j];
        const float avg_length = (l.length + r.length) / 2;
        const float angle_diff = std::acos(std::clamp(std::abs(l.axis.dot(r.axis)), 0.f, 1.f)) * static_cast<float>(180.0 / CV_PI);
        const float length_ratio = std::max(l.length, r.length) / std::min(l.length, r.length);

        cv::Point2f vertical = l.axis + r.axis;                         // 装甲板竖直轴
        const float axis_norm = static_cast<float>(cv::norm(vertical)); // 求vertical的模
        vertical *= 1.f / axis_norm;                                    // 将vertical单位化
        const cv::Point2f horizontal(vertical.y, -vertical.x);          // 装甲板水平向量
        const cv::Point2f delta = light_bars[order[j]].center - light_bars[order[i]].center; // 灯条中心的位移
        const float width = std::abs(delta.dot(horizontal)) / avg_length; // 标准化装甲板宽度
        const float y_offset = std::abs(delta.dot(vertical)) / avg_length; // 标准化装甲板上下错位

        if (angle_diff > max_angle_diff || length_ratio > max_length_ratio || y_offset > max_y_diff_ratio || width < min_aspect_ratio || width > max_aspect_ratio) // 硬阈值过滤
            continue;

        const double wide_penalty = std::max(0.f, width - 2.5f) / .6;                          // 过宽惩罚
        const double shape_penalty = .5 * (l.width / l.length + r.width / r.length);           // 形状惩罚
        const double cost = .35 * std::pow(angle_diff / max_angle_diff, 2)                     // 角度差惩罚
                            + .8 * std::pow(std::log(length_ratio), 2)                         // 长度一致性惩罚
                            + 2.0 * std::pow(y_offset, 2)                                      // 上下错位惩罚
                            + wide_penalty * wide_penalty + shape_penalty;
        const double score = std::exp(-cost) * std::sqrt(light_quality[order[i]] * light_quality[order[j]]); // 装甲板得分

        if (score < min_detection_score) // 硬阈值过滤
            continue;
        candidates.push_back({i, j, score});
    }

    // 贪心算法，根据分数进行排序
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b)
              {
                  if (a.score != b.score)
                      return a.score > b.score;
                  if (a.left != b.left)
                      return a.left < b.left;
                  return a.right < b.right;
              });

    std::vector<::armor> armors;

    // 去重，一根灯条只能匹配一个装甲板
    std::vector<bool> used(order.size(), false);
    for (const auto &c : candidates)
    {
        if (used[c.left] || used[c.right])
            continue;
        auto &armor = armors.emplace_back();
        armor.center = (light_bars[order[c.left]].center + light_bars[order[c.right]].center) / 2.f;
        armor.vertices[0] = lights[c.left].top;
        armor.vertices[1] = lights[c.left].bottom;
        armor.vertices[2] = lights[c.right].bottom;
        armor.vertices[3] = lights[c.right].top;
        used[c.left] = used[c.right] = true;
    }

    // 将装甲板从左往右排序
    std::sort(armors.begin(), armors.end(), [](const ::armor &a, const ::armor &b)
              { return a.center.x < b.center.x; });
    return armors;
}


// 绘制装甲板矩形
void detector::draw_armors(cv::Mat &image, const std::vector<::armor> &armors)
{
    for (const auto &armor : armors)
    {
        for (int i = 0; i < 4; i++)
        {
            cv::line(image, armor.vertices[i], armor.vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
}

// 绘制图形化UI
void detector::draw_video_info(cv::Mat &image, const std::vector<::armor> &armors, const std::string &status)
{
    std::vector<std::string> lines{status, "Camera XYZ (m): X right, Y down, Z forward"};
    if (armors.empty())
        lines.push_back("No armor detected");
    for (int i = 0; i < static_cast<int>(armors.size()); ++i)
    {
        const auto &armor = armors[i];
        const auto label = "#" + std::to_string(i + 1);
        lines.push_back(cv::format("%s tvec (m): [%.3f, %.3f, %.3f]", label.c_str(),
                                   armor.tvec.at<double>(0), armor.tvec.at<double>(1), armor.tvec.at<double>(2)));
        lines.push_back(cv::format("%s rvec (rad): [%.3f, %.3f, %.3f]", label.c_str(),
                                   armor.rvec.at<double>(0), armor.rvec.at<double>(1), armor.rvec.at<double>(2)));
        cv::putText(image, label, armor.center, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 0, 0}, 4, cv::LINE_AA);
        cv::putText(image, label, armor.center, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 255, 255}, 2, cv::LINE_AA);
    }
    for (int i = 0; i < static_cast<int>(lines.size()); ++i)
    {
        const cv::Point position(20, 30 + static_cast<int>(i) * 28);
        cv::putText(image, lines[i], position, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 0, 0}, 4, cv::LINE_AA);
        cv::putText(image, lines[i], position, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 255, 255}, 2, cv::LINE_AA);
    }
}

std::vector<::armor> detector::detect(const cv::Mat &image, EnemyColor color)
{
    // 先用高阈值检测，无匹配时再降低阈值
    const int thresholds[][2] = {{70, 170}, {40, 100}};
    for (const auto &threshold : thresholds)
    {
        const auto mask = detector::extract_color(image, color, threshold[0], threshold[1]);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        std::vector<float> quality;
        const auto lights = detector::get_valid_light_rects(contours, quality);
        auto armors = detector::match_armors(lights, quality);
        if (!armors.empty())
            return armors;
    }
    return {};
}
