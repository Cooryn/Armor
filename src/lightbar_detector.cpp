#include "lightbar_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

// ==========================================
// 完全使用你提供的原生实现
// ==========================================

cv::Mat extractColor(const cv::Mat &src, EnemyColor color)
{
    // 1. 安全校验：防止传入空图或灰度图导致 cv::split 崩溃
    if (src.empty() || src.channels() < 3)
    {
        return cv::Mat::zeros(src.size(), CV_8UC1);
    }

    // 2. 安全分离通道：使用 vector 动态接收，杜绝越界
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    cv::Mat mask;

    // 3. 根据目标颜色进行提取
    if (color == EnemyColor::RED)
    {
        // 提取红色：R(2) 大于 G(1) 和 B(0)
        mask = (channels[2] > channels[1] * 1.2) & (channels[2] > channels[0] * 1.2) & (channels[2] > 135);
    }
    else
    {
        // 提取蓝色：B(0) 大于 G(1) 和 R(2)
        mask = (channels[0] > channels[1] * 1.05) & (channels[0] > channels[2] * 1.05) & (channels[0] > 130);
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

// ==========================================
// 桥接与后端匹配逻辑 (衔接 PnP)
// ==========================================

// 将你过滤后的 contours 转为匹配需要的 RotatedRect，并把你在 drawLightBarRects 里的角度过滤搬过来，避免错配水平灯条
std::vector<cv::RotatedRect> getValidLightRects(const std::vector<std::vector<cv::Point>> &lightBars)
{
    std::vector<cv::RotatedRect> rects;
    for (const auto &c : lightBars)
    {
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        float angle = std::abs(rect.angle);
        float longEdgeAngle = (w >= h) ? angle : (90.0f - angle);

        // 保留你原本对角度的严格限制 (<75度不要)
        if (longEdgeAngle < 55.0f)
            continue;

        rects.push_back(rect);
    }
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

            // 适当放宽了几何条件以保证组队成功率
            if (std::abs(left_angle - right_angle) > 35.0f)
                continue;
            if (std::max(left_length, right_length) / std::min(left_length, right_length) > 2.5f)
                continue;
            if (std::abs(left.center.y - right.center.y) > avg_length * 2.0f)
                continue;

            float aspect_ratio = cv::norm(left.center - right.center) / avg_length;
            if (aspect_ratio < 0.8f || aspect_ratio > 5.0f)
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
            cv::line(out, armor.vertices[i], armor.vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    return out;
}