#include "lightbar_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

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
    const std::vector<std::vector<cv::Point>> &lightBars, float min_angle)
{
    std::vector<cv::RotatedRect> rects;
    for (const auto &c : lightBars)
    {
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        float angle = std::abs(rect.angle);
        float longEdgeAngle = (w >= h) ? angle : (90.0f - angle);

        if (longEdgeAngle < min_angle)
            continue;

        rects.push_back(rect);
    }
    return rects;
}

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat maskCopy = mask.clone();
    cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

std::vector<std::vector<cv::Point>> filterLightBars(const std::vector<std::vector<cv::Point>> &contours, double minAspectRatio, double minArea)
{
    std::vector<std::vector<cv::Point>> result;
    for (const auto &c : contours)
    {
        double area = cv::contourArea(c);
        if (area < minArea)
            continue;

        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        if (w < 1.0f || h < 1.0f)
            continue;

        float aspect = std::max(w, h) / std::min(w, h);
        if (aspect >= minAspectRatio)
        {
            result.push_back(c);
        }
    }
    return result;
}

cv::Mat drawLightBarRects(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &lightBars)
{
    cv::Mat out = src.clone();
    for (const auto &c : lightBars)
    {
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        float angle = std::abs(rect.angle);
        float longEdgeAngle = (w >= h) ? angle : (90.0f - angle);
        if (longEdgeAngle < 75.0f)
            continue;

        cv::Point2f vertices[4];
        rect.points(vertices);
        for (int j = 0; j < 4; j++)
        {
            cv::line(out, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 255), 2);
        }
    }
    return out;
}

std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars,
                              float max_angle_diff,
                              float max_length_ratio,
                              float min_aspect_ratio,
                              float max_y_diff_ratio)
{
    std::vector<Armor> armors;
    if (lightBars.size() < 2)
        return armors;

    std::vector<cv::RotatedRect> sortedBars = lightBars;

    std::sort(sortedBars.begin(), sortedBars.end(), [](const cv::RotatedRect &a, const cv::RotatedRect &b)
              { return a.center.x < b.center.x; });

    std::vector<bool> used(sortedBars.size(), false);

    for (size_t i = 0; i < sortedBars.size() - 1; i++)
    {
        if (used[i])
            continue;

        for (size_t j = i + 1; j < sortedBars.size(); j++)
        {
            if (used[j])
                continue;

            const auto &left = sortedBars[i];
            const auto &right = sortedBars[j];

            float left_length = std::max(left.size.width, left.size.height);
            float right_length = std::max(right.size.width, right.size.height);
            float avg_length = (left_length + right_length) / 2.0f;
            float left_angle = left.size.width > left.size.height ? left.angle : left.angle - 90.0f;
            float right_angle = right.size.width > right.size.height ? right.angle : right.angle - 90.0f;

            float angle_diff = std::abs(left_angle - right_angle);

            // 1. 替换角度差限制
            if (angle_diff > max_angle_diff && std::abs(angle_diff - 180.0f) > max_angle_diff)
                continue;

            // 2. 替换长短比限制
            if (std::max(left_length, right_length) / std::min(left_length, right_length) > max_length_ratio)
                continue;

            // 3. 替换 Y 轴错位限制
            if (std::abs(left.center.y - right.center.y) > avg_length * max_y_diff_ratio)
                continue;

            float aspect_ratio = cv::norm(left.center - right.center) / avg_length;

            // 4. 替换宽高比限制 (注意上限 4.5f 通常不需要调，所以保留硬编码)
            if (aspect_ratio < min_aspect_ratio || aspect_ratio > 4.5f)
                continue;

            Armor armor;
            armor.left_light = left;
            armor.right_light = right;
            armor.center = (left.center + right.center) / 2.0f;

            cv::Point2f left_pts[4], right_pts[4];
            left.points(left_pts);
            right.points(right_pts);

            std::sort(left_pts, left_pts + 4, [](const cv::Point2f &a, const cv::Point2f &b)
                      { return a.y < b.y; });
            std::sort(right_pts, right_pts + 4, [](const cv::Point2f &a, const cv::Point2f &b)
                      { return a.y < b.y; });

            armor.vertices[0] = (left_pts[0] + left_pts[1]) / 2.0f;   // 0: 左上 (Top-Left)
            armor.vertices[1] = (left_pts[2] + left_pts[3]) / 2.0f;   // 1: 左下 (Bottom-Left)
            armor.vertices[2] = (right_pts[2] + right_pts[3]) / 2.0f; // 2: 右下 (Bottom-Right)
            armor.vertices[3] = (right_pts[0] + right_pts[1]) / 2.0f; // 3: 右上 (Top-Right)

            armors.push_back(armor);

            used[i] = true;
            used[j] = true;
            break;
        }
    }
    return armors;
}

cv::Mat drawArmors(const cv::Mat &src, const std::vector<Armor> &armors)
{
    cv::Mat out = src.clone();
    for (const auto &armor : armors)
    {
        for (int i = 0; i < 4; i++)
        {
            cv::line(out, armor.vertices[i], armor.vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    return out;
}