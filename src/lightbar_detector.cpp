#include "lightbar_detector.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

// 提取灯条mask
cv::Mat extractColor(const cv::Mat &image, EnemyColor color, int color_th, int gray_th)
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
std::vector<cv::RotatedRect> getValidLightRects(
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
        if (w < 1.f || h < 1.f || std::max(w, h) / std::min(w, h) < minAspectRatio ||
            area / (w * h) < .30)
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
        const float fill = static_cast<float>(area) / rect.size.area(); // 计算填充率
        quality.push_back(std::sqrt(fit_quality * std::clamp(fill / .75f, .5f, 1.f))); // 计算灯条质量
        rects.push_back(rect);
    }
    return rects;
}

// 重定义灯条信息
struct LightGeometry
{
    float length, width;
    cv::Point2f axis;
    cv::Point2f top, bottom;
};

// 定义候选装甲板
struct Candidate
{
    size_t left, right;
    double score;
};

// 灯条匹配装甲板
std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &light_bars,
                               const std::vector<float> &light_quality,
                               float max_angle_diff,
                               float max_length_ratio,
                               float min_aspect_ratio,
                               float max_y_diff_ratio,
                               float max_aspect_ratio,
                               float min_detection_score)
{
    std::vector<Armor> armors;

    // 按照灯条中心位置从左往右、从上到下排序，将顺序存入order
    std::vector<cv::RotatedRect> bars;
    bars.reserve(light_bars.size());
    std::vector<size_t> order(light_bars.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
              {
        if (light_bars[a].center.x != light_bars[b].center.x)
            return light_bars[a].center.x < light_bars[b].center.x;
        return light_bars[a].center.y < light_bars[b].center.y; });

    // 将light_quality按照排序后的顺序存入qualities
    std::vector<float> qualities;
    for (size_t i = 0; i < order.size(); ++i)
    {
        bars.push_back(light_bars[order[i]]);
        qualities.push_back(light_quality[order[i]]);
    }

    // 初始化LightGeometry对象
    std::vector<LightGeometry> lights;
    for (const auto &bar : bars)
    {
        const float angle = (bar.angle + (bar.size.width < bar.size.height ? 90.f : 0.f)) * static_cast<float>(CV_PI / 180.0);
        cv::Point2f axis(std::cos(angle), std::sin(angle));
        if (axis.y < 0)
            axis *= -1.f;
        const float length = std::max(bar.size.width, bar.size.height);
        lights.push_back({length, std::min(bar.size.width, bar.size.height), axis,
                          bar.center - axis * (length / 2), bar.center + axis * (length / 2)});
    }
    std::vector<Candidate> candidates;

    // 枚举所有灯条组合
    for (size_t i = 0; i < bars.size(); ++i)
    {
        for (size_t j = i + 1; j < bars.size(); ++j)
        {
            const auto &l = lights[i];
            const auto &r = lights[j];
            const float avg_length = (l.length + r.length) / 2;
            const float angle_diff = std::acos(std::clamp(std::abs(l.axis.dot(r.axis)), 0.f, 1.f)) * static_cast<float>(180.0 / CV_PI);
            const float length_ratio = std::max(l.length, r.length) / std::min(l.length, r.length);
            cv::Point2f vertical = l.axis + r.axis;
            const float axis_norm = static_cast<float>(cv::norm(vertical)); // 求vertical的模
            vertical *= 1.f / axis_norm;                               // 将vertical单位化
            const cv::Point2f horizontal(vertical.y, -vertical.x);     // 装甲板水平向量
            const cv::Point2f delta = bars[j].center - bars[i].center; // 灯条中心的位移
            // 均在装甲板的局部坐标系中测量
            const float width = std::abs(delta.dot(horizontal)) / avg_length;
            const float y_offset = std::abs(delta.dot(vertical)) / avg_length;
            if (angle_diff > max_angle_diff || length_ratio > max_length_ratio ||
                y_offset > max_y_diff_ratio || width < min_aspect_ratio || width > max_aspect_ratio) // 硬阈值过滤
                continue;

            int intervening = 0; // 检查两根灯条中间有无其他灯条
            for (size_t k = i + 1; k < j; ++k)
            {
                const float offset = std::abs((bars[k].center - bars[i].center).dot(vertical));
                if (offset < avg_length * .5f && lights[k].length > avg_length * .5f)
                    ++intervening;
            }
            // 允许width缩短，但过宽的配对和偏斜的端点会增加惩罚
            const double wide_penalty = std::max(0.f, width - 2.5f) / .6;                          // 宽度惩罚
            const double shape_penalty = .5 * (l.width / l.length + r.width / r.length);           // 形状惩罚
            const double cost = .35 * std::pow(angle_diff / max_angle_diff, 2)                     // 角度差惩罚
                                + .8 * std::pow(std::log(length_ratio), 2)                         // 长度一致性惩罚
                                + 2.0 * std::pow(y_offset, 2)                                      // 上下错位惩罚
                                + wide_penalty * wide_penalty + shape_penalty + 1.5 * intervening; // 总惩罚
            const double score = std::exp(-cost) * std::sqrt(qualities[i] * qualities[j]);         // 装甲板得分
            if (score < min_detection_score)
                continue;
            candidates.push_back({i, j, score});
        }
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

    // 去重，一根灯条只能匹配一个装甲板
    std::vector<bool> used(bars.size(), false);
    for (const auto &c : candidates)
    {
        if (used[c.left] || used[c.right])
            continue;
        auto &armor = armors.emplace_back();
        armor.left_light = bars[c.left];
        armor.right_light = bars[c.right];
        armor.center = (bars[c.left].center + bars[c.right].center) / 2.f;
        armor.vertices[0] = lights[c.left].top;
        armor.vertices[1] = lights[c.left].bottom;
        armor.vertices[2] = lights[c.right].bottom;
        armor.vertices[3] = lights[c.right].top;
        armor.detection_score = c.score;
        used[c.left] = used[c.right] = true;
    }

    // 将装甲板从左往右排序
    std::sort(armors.begin(), armors.end(), [](const Armor &a, const Armor &b)
              { return a.center.x < b.center.x; });
    return armors;
}

void drawArmors(cv::Mat &image, const std::vector<Armor> &armors)
{
    for (const auto &armor : armors)
    {
        for (int i = 0; i < 4; i++)
        {
            cv::line(image, armor.vertices[i], armor.vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
}

void drawVideoInfo(cv::Mat &image, const std::vector<Armor> &armors, const std::string &status)
{
    std::vector<std::string> lines{status, "Camera XYZ (m): X right, Y down, Z forward"};
    if (armors.empty())
        lines.push_back("No armor detected");
    for (std::size_t i = 0; i < armors.size(); ++i)
    {
        const auto &armor = armors[i];
        const auto label = "#" + std::to_string(i + 1);
        lines.push_back(cv::format("%s  X=%.3f  Y=%.3f  Z=%.3f", label.c_str(),
            armor.tvec.at<double>(0), armor.tvec.at<double>(1), armor.tvec.at<double>(2)));
        cv::putText(image, label, armor.center, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 0, 0}, 4, cv::LINE_AA);
        cv::putText(image, label, armor.center, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 255, 255}, 2, cv::LINE_AA);
    }
    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        const cv::Point position(20, 30 + static_cast<int>(i) * 28);
        cv::putText(image, lines[i], position, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 0, 0}, 4, cv::LINE_AA);
        cv::putText(image, lines[i], position, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 255, 255}, 2, cv::LINE_AA);
    }
}

std::vector<Armor> detectArmors(const cv::Mat &image, EnemyColor color)
{
    const auto mask = extractColor(image, color, 70, 170);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // 提取灯条轮廓
    std::vector<float> quality;
    const auto lights = getValidLightRects(contours, quality, 55, 1.5, 40);
    return matchArmors(lights, quality, 20, 2.f, .8f, .8f, 3.1f, .35f);
}
