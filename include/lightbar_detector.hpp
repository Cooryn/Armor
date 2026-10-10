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

class detector
{
public:
    static std::vector<::armor> detect(const cv::Mat &image, EnemyColor color);
    static cv::Mat extract_color(const cv::Mat &image, EnemyColor color, int color_th = 20, int gray_th = 80);
    static std::vector<cv::RotatedRect> get_valid_light_rects(
        const std::vector<std::vector<cv::Point>> &light_bars, std::vector<float> &quality,
        float min_angle = 55.0f, double minAspectRatio = 1.5, double minArea = 10.0);
    static std::vector<::armor> match_armors(
        const std::vector<cv::RotatedRect> &light_bars, const std::vector<float> &light_quality,
        float max_angle_diff = 20.0f, float max_length_ratio = 1.5f, float min_aspect_ratio = 1.2f,
        float max_y_diff_ratio = 0.8f, float max_aspect_ratio = 3.1f, float min_detection_score = 0.35f);
    static void draw_armors(cv::Mat &image, const std::vector<::armor> &armors);
    static void draw_video_info(cv::Mat &image, const std::vector<::armor> &armors, const std::string &status);
};
