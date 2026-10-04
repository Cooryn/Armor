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
struct Solver
{
    cv::Mat camera_matrix, distort_coeffs;
    std::vector<SolverTrack> previous;
    double previous_time = std::numeric_limits<double>::quiet_NaN();
};

SolvedFrame solver_solve_frame(Solver &self, std::vector<Armor> armors, double timestamp_ms,
                               const std::string &video_stem);

bool solver_solve(Solver &self, Armor &armor, double yaw_hint = std::numeric_limits<double>::quiet_NaN());
std::vector<double> solver_yaw_hints(const Solver &self, const std::vector<Armor> &armors, double timestamp);
void solver_finish_frame(Solver &self, const std::vector<Armor> &armors, double timestamp);
