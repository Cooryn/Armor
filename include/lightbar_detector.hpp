#pragma once
#include <opencv2/core.hpp>
#include <vector>

enum class EnemyColor
{
    RED,
    BLUE
};

struct Armor
{
    cv::RotatedRect left_light;
    cv::RotatedRect right_light;
    cv::Point2f center;
    cv::Point2f vertices[4]; // 顺时针顺序: TL(左上), TR(右上), BR(右下), BL(左下)

    cv::Mat rvec; // PnP 解算出的旋转向量
    cv::Mat tvec; // PnP 解算出的平移向量 (X, Y, Z 距离)
    double yaw;   // PnP 解算出的偏航角
};

cv::Mat extractColor(const cv::Mat &src, EnemyColor color);
std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask);
std::vector<std::vector<cv::Point>> filterLightBars(
    const std::vector<std::vector<cv::Point>> &contours, double minAspectRatio = 2.0, double minArea = 30.0);
std::vector<cv::RotatedRect> getLightBarRects(const std::vector<std::vector<cv::Point>> &contours);
std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars);
cv::Mat drawArmors(const cv::Mat &src, const std::vector<Armor> &armors);