#pragma once
#include <opencv2/opencv.hpp>
#include "lightbar_detector.hpp"
#include <limits>

class Solver
{
private:
    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;

    std::vector<cv::Point3f> object_points;
    struct Track {
        cv::Point2f center, velocity;
        double width, yaw, yaw_rate;
    };
    std::vector<Track> previous;
    double previous_time = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> temporalMatches(const std::vector<Armor> &armors, double timestamp) const;

public:
    Solver(const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs);

    bool solve(Armor &armor, double yaw_hint = std::numeric_limits<double>::quiet_NaN());
    std::vector<double> yawHints(const std::vector<Armor> &armors, double timestamp) const;
    void finishFrame(const std::vector<Armor> &armors, double timestamp);
};
