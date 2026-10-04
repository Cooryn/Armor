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
    Solver() = default;
    Solver(const cv::Mat &camera, const cv::Mat &distortion);
    SolvedFrame solve_frame(std::vector<Armor> armors, double timestamp_ms, const std::string &video_stem);
    bool solve(Armor &armor, double yaw_hint = std::numeric_limits<double>::quiet_NaN());
    std::vector<double> yaw_hints(const std::vector<Armor> &armors, double timestamp) const;
    void finish_frame(const std::vector<Armor> &armors, double timestamp);

    cv::Mat camera_matrix, distort_coeffs;

private:
    std::vector<SolverTrack> previous;
    double previous_time = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> temporal_matches(const std::vector<Armor> &armors, double timestamp) const;
};
