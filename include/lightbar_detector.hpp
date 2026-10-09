#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

enum EnemyColor
{
    ENEMY_RED,
    ENEMY_BLUE
};

cv::Mat extractColor(const cv::Mat &src, EnemyColor color, int color_th = 20, int gray_th = 80);

struct Armor
{
    cv::RotatedRect left_light;
    cv::RotatedRect right_light;
    cv::Point2f center;
    cv::Point2f vertices[4];
    cv::Mat rvec;
    cv::Mat tvec;
    double yaw = 0.0;
    double detection_score = 0.0;
    double reprojection_error = 0.0;
    int pnp_candidate_count = 0;
    bool pnp_used_temporal = false;
};

std::vector<cv::RotatedRect> getValidLightRects(
    const std::vector<std::vector<cv::Point>> &lightBars, std::vector<float> &quality,
    float min_angle = 55.0f,
    double minAspectRatio = 1.5, double minArea = 10.0);

std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars,
                               const std::vector<float> &light_quality,
                               float max_angle_diff = 20.0f,
                               float max_length_ratio = 1.5f,
                               float min_aspect_ratio = 1.2f,
                               float max_y_diff_ratio = 0.8f,
                               float max_aspect_ratio = 3.1f,
                               float min_detection_score = 0.35f);
void drawArmors(cv::Mat &src, const std::vector<Armor> &armors);
void drawVideoInfo(cv::Mat &image, const std::vector<Armor> &armors, const std::string &status);

std::vector<Armor> detectArmors(const cv::Mat &image, EnemyColor color);
