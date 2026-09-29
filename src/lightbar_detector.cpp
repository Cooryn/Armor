#include "lightbar_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

cv::Mat extractColor(const cv::Mat &src, EnemyColor color, int color_th, int gray_th)
{
    if (src.empty() || src.channels() < 3)
        return cv::Mat::zeros(src.size(), CV_8UC1);

    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    cv::Mat color_mask;

    if (color == EnemyColor::RED)
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

    cv::Mat gray, highlight_mask;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, highlight_mask, 210, 255, cv::THRESH_BINARY);

    cv::Mat shield;
    cv::Mat big_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::dilate(color_mask, shield, big_kernel);

    cv::bitwise_and(highlight_mask, shield, highlight_mask);

    cv::bitwise_or(color_mask, highlight_mask, color_mask);

    cv::Mat small_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, small_kernel);

    return color_mask;
}

std::vector<cv::RotatedRect> getValidLightRects(
    const std::vector<std::vector<cv::Point>> &lightBars, float min_angle,
    std::vector<float> *quality,
    double minAspectRatio, double minArea)
{
    std::vector<cv::RotatedRect> rects;
    if (quality) quality->clear();
    for (const auto &c : lightBars)
    {
        if (c.size() < 3) continue;
        const double area = cv::contourArea(c);
        if (area <= 0 || area < minArea) continue;
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        if (w < 1.f || h < 1.f || std::max(w, h) / std::min(w, h) < minAspectRatio ||
            area / (w * h) < .30) continue;
        float angle = std::abs(rect.angle);
        float longEdgeAngle = (w >= h) ? angle : (90.0f - angle);

        if (longEdgeAngle < min_angle)
            continue;

        // Use the full pixel contour directly, without resampling.
        const std::vector<cv::Point2f> boundary(c.begin(), c.end());
        cv::Vec4f line;
        cv::fitLine(boundary, line, cv::DIST_HUBER, 0, .01, .01);
        cv::Point2f axis(line[0], line[1]);
        if (axis.y < 0) axis *= -1.f;
        const cv::Point2f origin(line[2], line[3]);
        float fitted_angle = std::atan2(axis.y, axis.x) * static_cast<float>(180.0 / CV_PI);
        const float old_angle = rect.angle + (w < h ? 90.f : 0.f);
        const float axis_change = std::abs(std::remainder(fitted_angle-old_angle, 180.f));
        float fit_quality = .75f;
        // Broad/irregular blobs do not have a reliably identifiable long axis.
        if (std::max(w,h) >= 2.f * std::min(w,h) && axis_change <= 10.f) {
            // Blend rather than replace the bounding-box axis: short rasterized
            // contours cannot support an arbitrarily precise line orientation.
            fitted_angle = old_angle + .5f * std::remainder(fitted_angle-old_angle, 180.f);
            const float radians = fitted_angle * static_cast<float>(CV_PI / 180.0);
            axis = cv::Point2f(std::cos(radians), std::sin(radians));
            if (axis.y < 0) axis *= -1.f;
            const cv::Point2f normal(axis.y, -axis.x);
            float along_min = std::numeric_limits<float>::infinity();
            float along_max = -std::numeric_limits<float>::infinity();
            std::vector<float> across;
            for (const auto &point : boundary) {
                const float projection = (point-origin).dot(axis);
                along_min = std::min(along_min, projection);
                along_max = std::max(along_max, projection);
                across.push_back((point-origin).dot(normal));
            }
            std::sort(across.begin(), across.end());
            // Preserve the full measured length: trimming both ends biases depth.
            const float length = along_max-along_min;
            const float width = across.back()-across.front();
            const float offset = (across[across.size()/10] + across[across.size()*9/10]) / 2;
            const cv::Point2f center = origin + axis*((along_min+along_max)/2)
                                      + normal*offset;
            rect = cv::RotatedRect(center, cv::Size2f(length, width), fitted_angle);
            fit_quality = 1.f;
        }
        const float fill = static_cast<float>(area) /
                           std::max(1.f, rect.size.area());
        // A bounded confidence penalty for incomplete/irregular light shapes.
        // Oblique plate views are not penalized just for their screen angle.
        if (quality) quality->push_back(std::sqrt(fit_quality * std::clamp(fill/.75f, .5f, 1.f)));
        rects.push_back(rect);
    }
    return rects;
}

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    return contours;
}

namespace {
struct LightGeometry {
    float length, width;
    cv::Point2f axis;  // Unit long axis, pointing down in the image.
    cv::Point2f top, bottom;
};
LightGeometry geometry(const cv::RotatedRect &rect) {
    const float angle = (rect.angle + (rect.size.width < rect.size.height ? 90.f : 0.f))
                        * static_cast<float>(CV_PI / 180.0);
    cv::Point2f axis(std::cos(angle), std::sin(angle));
    if (axis.y < 0) axis *= -1.f;
    const float length = std::max(rect.size.width, rect.size.height);
    return {length, std::min(rect.size.width, rect.size.height), axis,
            rect.center - axis * (length / 2), rect.center + axis * (length / 2)};
}
struct Candidate { size_t left, right; Armor armor; };
}

std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars,
                              float max_angle_diff,
                              float max_length_ratio,
                              float min_aspect_ratio,
                              float max_y_diff_ratio,
                              float max_aspect_ratio,
                              float min_detection_score,
                              const std::vector<float> &light_quality)
{
    std::vector<Armor> armors;
    if (lightBars.size() < 2 || max_angle_diff <= 0 || max_length_ratio < 1 ||
        max_y_diff_ratio <= 0 || min_aspect_ratio <= 0 || max_aspect_ratio < min_aspect_ratio ||
        min_detection_score <= 0 || min_detection_score > 1)
        return armors;
    auto bars = lightBars;
    std::vector<size_t> order(bars.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (lightBars[a].center.x != lightBars[b].center.x)
            return lightBars[a].center.x < lightBars[b].center.x;
        return lightBars[a].center.y < lightBars[b].center.y;
    }); // 按照灯条中心位置从左往右、从上到下排序，将顺序存入order
    std::vector<float> qualities;
    for (size_t i = 0; i < order.size(); ++i) {
        bars[i] = lightBars[order[i]];
        const float q = light_quality.size() == bars.size() ? light_quality[order[i]] : 1.f;
        qualities.push_back(std::isfinite(q) ? std::clamp(q, .1f, 1.f) : .1f);
    } // 将light_quality按照排序后的顺序存入qualities

    std::vector<LightGeometry> lights;
    for (const auto &bar : bars)
        lights.push_back(geometry(bar)); // 初始化LightGeometry对象
    std::vector<Candidate> candidates;

    for (size_t i = 0; i < bars.size(); ++i) {
        for (size_t j = i + 1; j < bars.size(); ++j)
        { // 枚举所有灯条组合
            const auto &l = lights[i];
            const auto &r = lights[j];
            if (l.length < 1 || r.length < 1 || l.width <= 0 || r.width <= 0) continue;
            const float avg_length = (l.length + r.length) / 2;
            const float angle_diff = std::acos(std::clamp(std::abs(l.axis.dot(r.axis)), 0.f, 1.f))
                                     * static_cast<float>(180.0 / CV_PI);
            const float length_ratio = std::max(l.length, r.length) / std::min(l.length, r.length);
            cv::Point2f vertical = l.axis + r.axis;
            const float axis_norm = static_cast<float>(cv::norm(vertical)); // 求vertical的模
            if (axis_norm < 1e-6f) continue;
            vertical *= 1.f / axis_norm; // 将vertical单位化
            const cv::Point2f horizontal(vertical.y, -vertical.x); // 装甲板水平向量
            const cv::Point2f delta = bars[j].center - bars[i].center; // 灯条中心的位移
            // 均在装甲板的局部坐标系中测量
            const float width = std::abs(delta.dot(horizontal)) / avg_length;
            const float y_offset = std::abs(delta.dot(vertical)) / avg_length;
            if (angle_diff > max_angle_diff || length_ratio > max_length_ratio ||
                y_offset > max_y_diff_ratio || width < min_aspect_ratio || width > max_aspect_ratio) // 硬阈值过滤
                continue;

            int intervening = 0; // 检查两根灯条中间有无其他灯条
            for (size_t k = i + 1; k < j; ++k) {
                const float offset = std::abs((bars[k].center - bars[i].center).dot(vertical));
                if (offset < avg_length * .5f && lights[k].length > avg_length * .5f)
                    ++intervening;
            }
            // 允许width缩短，但过宽的配对和偏斜的端点会增加惩罚
            const double wide_penalty = std::max(0.f, width - 2.5f) / .6; // 宽度惩罚
            const double shape_penalty = .5 * (l.width / l.length + r.width / r.length); // 形状惩罚
            const double cost = .35 * std::pow(angle_diff / max_angle_diff, 2) // 角度差惩罚
                              + .8 * std::pow(std::log(length_ratio), 2) // 长度一致性惩罚
                              + 2.0 * std::pow(y_offset, 2) // 上下错位惩罚
                              + wide_penalty * wide_penalty + shape_penalty + 1.5 * intervening; // 总惩罚
            const double score = std::exp(-cost) * std::sqrt(qualities[i]*qualities[j]); // 装甲板得分
            if (score < min_detection_score) continue;
            Armor armor; // 创建Armor
            armor.left_light = bars[i]; armor.right_light = bars[j];
            armor.center = (bars[i].center + bars[j].center) / 2.f;
            armor.vertices[0] = l.top; armor.vertices[1] = l.bottom;
            armor.vertices[2] = r.bottom; armor.vertices[3] = r.top;
            armor.detection_score = score;
            candidates.push_back({i, j, armor});
        }
    }

    // Prefer the highest-score pair; each light can belong to only one armor.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) {
        if (a.armor.detection_score != b.armor.detection_score)
            return a.armor.detection_score > b.armor.detection_score;
        if (a.left != b.left) return a.left < b.left;
        return a.right < b.right;
    });
    std::vector<bool> used(bars.size(), false);
    for (const auto &c : candidates) {
        if (used[c.left] || used[c.right]) continue;
        armors.push_back(c.armor);
        used[c.left] = used[c.right] = true;
    }
    std::sort(armors.begin(), armors.end(), [](const Armor &a, const Armor &b) {
        return a.center.x < b.center.x;
    });
    return armors;
}

void drawArmors(cv::Mat &src, const std::vector<Armor> &armors)
{
    for (const auto &armor : armors)
    {
        for (int i = 0; i < 4; i++)
        {
            cv::line(src, armor.vertices[i], armor.vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
}
