#pragma once
#include <opencv2/opencv.hpp>
#include "lightbar_detector.hpp"
#include <Eigen/Dense>
#include <limits>

struct SolvedFrame
{
    std::vector<::armor> armors;
    std::vector<Eigen::Vector4d> observations;
};
struct SolverTrack
{
    cv::Point2f center, velocity;
    double width, yaw, yaw_rate;
};
class solver
{
public:
    // 初始化相机内参和畸变系数
    solver(const cv::Mat &camera = cv::Mat(), const cv::Mat &distortion_coefficients = cv::Mat())
    {
        camera_matrix = camera;
        distort_coeffs = distortion_coefficients;
    }
    void use_video_profile(int profile);
    SolvedFrame solve_frame(std::vector<::armor> armors, double timestamp_ms);
    bool solve(::armor &armor, double yaw_hint = std::numeric_limits<double>::quiet_NaN());
    std::vector<double> yaw_hints(const std::vector<::armor> &armors, double timestamp_s) const;
    void finish_frame(const std::vector<::armor> &armors, double timestamp_s);

    cv::Mat camera_matrix, distort_coeffs;

private:
    std::vector<SolverTrack> previous;
    double previous_timestamp_s = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> temporal_matches(const std::vector<::armor> &armors, double timestamp_s) const;
};
