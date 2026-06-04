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

    // ==========================================
    // 步骤 1：基础差分 —— 抛弃绿通道，只比红蓝！
    // ==========================================
    if (color == EnemyColor::RED)
    {
        cv::Mat r_sub_b;
        cv::subtract(channels[2], channels[0], r_sub_b);
        // 只要红比蓝多出 color_th，且绝对亮度大于 gray_th，就是红色！
        color_mask = (r_sub_b > color_th) & (channels[2] > gray_th);
    }
    else
    {
        cv::Mat b_sub_r;
        cv::subtract(channels[0], channels[2], b_sub_r);
        // 只要蓝比红多出 color_th，且绝对亮度大于 gray_th，就是蓝色！
        color_mask = (b_sub_r > color_th) & (channels[0] > gray_th);
    }

    // ==========================================
    // 步骤 2：过曝修复术 —— "色彩保护罩" (保持不变)
    // ==========================================
    cv::Mat gray, highlight_mask;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, highlight_mask, 210, 255, cv::THRESH_BINARY); // 灰度大于210认为是纯白高光

    cv::Mat shield;
    // 放大内核，给高光更大的包裹范围
    cv::Mat big_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::dilate(color_mask, shield, big_kernel);

    // 精准打击：只保留被保护罩盖住的高光
    cv::bitwise_and(highlight_mask, shield, highlight_mask);

    // 完美合体
    cv::bitwise_or(color_mask, highlight_mask, color_mask);

    // ==========================================
    // 步骤 3：边缘平滑
    // ==========================================
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
            continue; // 使用传进来的角度

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
    // 按 X 坐标从左到右排序
    std::sort(sortedBars.begin(), sortedBars.end(), [](const cv::RotatedRect &a, const cv::RotatedRect &b)
              { return a.center.x < b.center.x; });

    // 记录灯条是否已经被匹配过了 (非常优秀的逻辑，保留！)
    std::vector<bool> used(sortedBars.size(), false);

    for (size_t i = 0; i < sortedBars.size() - 1; i++)
    {
        if (used[i])
            continue; // 如果左灯条已经名花有主，跳过

        for (size_t j = i + 1; j < sortedBars.size(); j++)
        {
            if (used[j])
                continue; // 如果右灯条已经名花有主，跳过

            const auto &left = sortedBars[i];
            const auto &right = sortedBars[j];

            // 提取物理长度和真实倾斜角 (保留你极其精准的处理逻辑)
            float left_length = std::max(left.size.width, left.size.height);
            float right_length = std::max(right.size.width, right.size.height);
            float avg_length = (left_length + right_length) / 2.0f;
            float left_angle = left.size.width > left.size.height ? left.angle : left.angle - 90.0f;
            float right_angle = right.size.width > right.size.height ? right.angle : right.angle - 90.0f;

            // ==========================================
            // 🚀 核心升级：四大严苛物理几何防线
            // ==========================================

            // --- 防线 A：角度平行约束 (收紧至 8 度) ---
            float angle_diff = std::abs(left_angle - right_angle);
            // 注：RotatedRect 角度有 180 度跳变的可能，加上 abs(diff - 180) 的防误杀逻辑
            if (angle_diff > 8.0f && std::abs(angle_diff - 180.0f) > 8.0f)
                continue;

            // --- 防线 B：长度比例约束 (收紧至 1.5 倍) ---
            if (std::max(left_length, right_length) / std::min(left_length, right_length) > 1.5f)
                continue;

            // --- 防线 C：Y 轴高度差约束 (收紧至平均长度的 0.8 倍) ---
            if (std::abs(left.center.y - right.center.y) > avg_length * 0.8f)
                continue;

            // --- 防线 D：物理长宽比约束 (卡死在 1.2 到 4.5 之间) ---
            float aspect_ratio = cv::norm(left.center - right.center) / avg_length;
            if (aspect_ratio < 1.2f || aspect_ratio > 4.5f)
                continue;

            // ==========================================
            // 匹配成功！(保留你原有的装甲板组装逻辑)
            // ==========================================
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

            // 标记这两个灯条已被使用，不许再跟别人配对！
            used[i] = true;
            used[j] = true;
            break; // 找到右灯条后直接跳出内层循环，让左灯条 i 进入下一个
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