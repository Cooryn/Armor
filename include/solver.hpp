#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "lightbar_detector.hpp"

class Solver
{
private:
    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;
    std::vector<cv::Point3f> object_points; // 3D模型点 (顺时针顺序)

public:
    Solver(const cv::Mat &camera_mat, const cv::Mat &dist_coeffs);
    bool solve(Armor &armor);
    void drawAxis(cv::Mat &image, const Armor &armor);
};