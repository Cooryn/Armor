#include "lightbar_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

cv::Mat extractColor(const cv::Mat &src, EnemyColor color)
{
    cv::Mat channels[3];
    cv::split(src, channels);
    cv::Mat mask;
    if (color == EnemyColor::RED)
    {
        mask = (channels[2] > channels[1] * 1.2) & (channels[2] > channels[0] * 1.2) & (channels[2] > 135);
    }
    else
    {
        mask = (channels[0] > channels[1] * 1.2) & (channels[0] > channels[2] * 1.2) & (channels[0] > 135);
    }
    return mask;
}

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

std::vector<std::vector<cv::Point>> filterLightBars(const std::vector<std::vector<cv::Point>> &contours, double minAspectRatio, double minArea)
{
    std::vector<std::vector<cv::Point>> result;
    for (const auto &c : contours)
    {
        if (cv::contourArea(c) < minArea)
            continue;
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width, h = rect.size.height;
        if (w < 1.0f || h < 1.0f)
            continue;
        if (std::max(w, h) / std::min(w, h) >= minAspectRatio)
            result.push_back(c);
    }
    return result;
}

std::vector<cv::RotatedRect> getLightBarRects(const std::vector<std::vector<cv::Point>> &contours)
{
    std::vector<cv::RotatedRect> rects;
    for (const auto &c : contours)
        rects.push_back(cv::minAreaRect(c));
    return rects;
}

std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars)
{
    std::vector<Armor> armors;
    if (lightBars.size() < 2)
        return armors;

    std::vector<cv::RotatedRect> sortedBars = lightBars;
    std::sort(sortedBars.begin(), sortedBars.end(), [](const cv::RotatedRect &a, const cv::RotatedRect &b)
              { return a.center.x < b.center.x; });

    for (size_t i = 0; i < sortedBars.size() - 1; i++)
    {
        for (size_t j = i + 1; j < sortedBars.size(); j++)
        {
            const auto &left = sortedBars[i];
            const auto &right = sortedBars[j];

            float left_length = std::max(left.size.width, left.size.height);
            float right_length = std::max(right.size.width, right.size.height);
            float avg_length = (left_length + right_length) / 2.0f;
            float left_angle = left.size.width > left.size.height ? left.angle : left.angle - 90.0f;
            float right_angle = right.size.width > right.size.height ? right.angle : right.angle - 90.0f;

            if (std::abs(left_angle - right_angle) > 15.0f)
                continue;
            if (std::max(left_length, right_length) / std::min(left_length, right_length) > 1.8f)
                continue;
            if (std::abs(left.center.y - right.center.y) > avg_length * 0.8f)
                continue;

            float aspect_ratio = cv::norm(left.center - right.center) / avg_length;
            if (aspect_ratio < 1.0f || aspect_ratio > 4.5f)
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

            // 顺时针角点 (TL, TR, BR, BL)
            armor.vertices[0] = (left_pts[0] + left_pts[1]) / 2.0f;
            armor.vertices[1] = (right_pts[0] + right_pts[1]) / 2.0f;
            armor.vertices[2] = (right_pts[2] + right_pts[3]) / 2.0f;
            armor.vertices[3] = (left_pts[2] + left_pts[3]) / 2.0f;

            armors.push_back(armor);
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
            cv::line(out, armor.vertices[i], armor.vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        }
        cv::circle(out, armor.center, 4, cv::Scalar(0, 255, 0), -1);
    }
    return out;
}