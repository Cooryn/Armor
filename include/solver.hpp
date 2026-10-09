#pragma once
#include <opencv2/opencv.hpp>
#include "lightbar_detector.hpp"
#include "predictor.hpp"
#include <limits>

struct SolvedFrame
{
    std::vector<Armor> armors;
    std::vector<VideoObservation> observations;
};
struct SolverTrack
{
    cv::Point2f center, velocity;
    double width, yaw, yaw_rate;
};
class Solver
{
public:
    // 初始化相机内参和畸变系数
    Solver(const cv::Mat &camera = cv::Mat(), const cv::Mat &distortion_coefficients = cv::Mat())
    {
        camera_matrix = camera;
        distort_coeffs = distortion_coefficients;
    }
    void use_video_profile(int profile);
    SolvedFrame solve_frame(std::vector<Armor> armors, double timestamp_ms);
    bool solve(Armor &armor, double yaw_hint = std::numeric_limits<double>::quiet_NaN());
    std::vector<double> yaw_hints(const std::vector<Armor> &armors, double timestamp_s) const;
    void finish_frame(const std::vector<Armor> &armors, double timestamp_s);

    cv::Mat camera_matrix, distort_coeffs;

private:
    std::vector<SolverTrack> previous;
    double previous_timestamp_s = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> temporal_matches(const std::vector<Armor> &armors, double timestamp_s) const;
};
