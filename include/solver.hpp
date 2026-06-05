#pragma once
#include <opencv2/opencv.hpp>
#include "lightbar_detector.hpp"

class Solver
{
private:
    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;

    std::vector<cv::Point3f> object_points;

public:
    Solver(const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs);

    bool solve(Armor &armor);
};