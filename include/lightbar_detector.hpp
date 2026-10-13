#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

enum EnemyColor
{
    ENEMY_RED,
    ENEMY_BLUE
};

struct armor
{
    cv::Point2f center;
    cv::Point2f vertices[4];
    cv::Mat rvec;
    cv::Mat tvec;
    double yaw = 0.0;
};

class detector
{
public:
    static std::vector<::armor> detect(const cv::Mat &image, EnemyColor color);
    static cv::Mat extract_color(const cv::Mat &image, EnemyColor color,
                                 int color_th = 40, int gray_th = 100);
    static std::vector<cv::RotatedRect> get_valid_light_rects(
        const std::vector<std::vector<cv::Point>> &light_bars, std::vector<float> &quality,
        float min_angle = 55.0f, double minAspectRatio = 1.5, double minArea = 12.0);
    static std::vector<::armor> match_armors(
        const std::vector<cv::RotatedRect> &light_bars, const std::vector<float> &light_quality,
        float max_angle_diff = 20.0f, float max_length_ratio = 2.0f, float min_aspect_ratio = .8f,
        float max_y_diff_ratio = .5f, float max_aspect_ratio = 3.1f,
        float min_detection_score = .4f);
    static void draw_armors(cv::Mat &image, const std::vector<::armor> &armors);
    static void draw_video_info(cv::Mat &image, const std::vector<::armor> &armors, const std::string &status);
};
