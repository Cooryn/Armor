#pragma once

#include <opencv2/core.hpp>
#include <vector>

enum class EnemyColor
{
    RED,
    BLUE
};

cv::Mat extractColor(const cv::Mat &src, EnemyColor color, int color_th = 20, int gray_th = 80);

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask);

std::vector<std::vector<cv::Point>> filterLightBars(
    const std::vector<std::vector<cv::Point>> &contours,
    double minAspectRatio = 1.5, double minArea = 10.0);

struct Armor
{
    cv::RotatedRect left_light;
    cv::RotatedRect right_light;
    cv::Point2f center;
    cv::Point2f vertices[4];
    cv::Mat rvec;
    cv::Mat tvec;
    double yaw = 0.0;
};

std::vector<cv::RotatedRect> getValidLightRects(
    const std::vector<std::vector<cv::Point>> &lightBars, float min_angle = 55.0f);

std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars);
cv::Mat drawArmors(const cv::Mat &src, const std::vector<Armor> &armors);