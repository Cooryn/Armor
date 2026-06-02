#include "lightbar_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>

// ==========================================
// 阶段一：灯条提取实现 (你的原始代码，未做删减)
// ==========================================
// 通用颜色提取函数
cv::Mat extractColor(const cv::Mat &src, EnemyColor color)
{
    cv::Mat channels[3];
    cv::split(src, channels); // channels[0]=B, channels[1]=G, channels[2]=R

    cv::Mat mask;

    if (color == EnemyColor::RED)
    {
        // 敌方是红色：判断 R 通道是否显著大于 G 和 B
        mask = (channels[2] > channels[1] * 1.2) & (channels[2] > channels[0] * 1.2) & (channels[2] > 135);
    }
    else if (color == EnemyColor::BLUE)
    {
        // 敌方是蓝色：判断 B 通道是否显著大于 G 和 R
        mask = (channels[0] > channels[1] * 1.2) & (channels[0] > channels[2] * 1.2) & (channels[0] > 135); // 蓝光的阈值可能需要根据摄像头实际曝光微调
    }

    return mask;
}

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat maskCopy = mask.clone();
    cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

cv::Mat applyMaskToImage(const cv::Mat &src, const cv::Mat &mask)
{
    cv::Mat result;
    cv::bitwise_and(src, src, result, mask);
    return result;
}

cv::Mat drawAllContours(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &contours)
{
    cv::Mat out = src.clone();
    cv::drawContours(out, contours, -1, cv::Scalar(0, 255, 0), 2);
    return out;
}

std::vector<std::vector<cv::Point>> filterLightBars(
    const std::vector<std::vector<cv::Point>> &contours,
    double minAspectRatio, double minArea)
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

// ==========================================
// 阶段二：装甲板匹配实现 (新增代码)
// ==========================================
std::vector<cv::RotatedRect> getLightBarRects(const std::vector<std::vector<cv::Point>> &contours)
{
    std::vector<cv::RotatedRect> rects;
    for (const auto &c : contours)
    {
        rects.push_back(cv::minAreaRect(c));
    }
    return rects;
}

std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars)
{
    std::vector<Armor> armors;
    if (lightBars.size() < 2)
        return armors;

    // 按 X 轴坐标从左到右排序
    std::vector<cv::RotatedRect> sortedBars = lightBars;
    std::sort(sortedBars.begin(), sortedBars.end(),
              [](const cv::RotatedRect &a, const cv::RotatedRect &b)
              {
                  return a.center.x < b.center.x;
              });

    // 双重循环，两两匹配
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

            float angle_diff = std::abs(left_angle - right_angle);
            float length_ratio = std::max(left_length, right_length) / std::min(left_length, right_length);
            float center_distance = cv::norm(left.center - right.center);
            float y_diff = std::abs(left.center.y - right.center.y);
            float aspect_ratio = center_distance / avg_length;

            // 几何阈值判断
            if (angle_diff > 15.0f)
                continue;
            if (length_ratio > 1.8f)
                continue;
            if (y_diff > avg_length * 0.8f)
                continue;
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

            // 获取四个角点 (顺序: 左上, 左下, 右下, 右上)
            armor.vertices[0] = (left_pts[0] + left_pts[1]) / 2.0f;
            armor.vertices[1] = (left_pts[2] + left_pts[3]) / 2.0f;
            armor.vertices[2] = (right_pts[2] + right_pts[3]) / 2.0f;
            armor.vertices[3] = (right_pts[0] + right_pts[1]) / 2.0f;

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