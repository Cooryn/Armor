#pragma once
#include <opencv2/opencv.hpp>
#include "lightbar_detector.hpp" // 确保包含了你的 Armor 结构体

class Solver
{
private:
    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;

    // 准备两套弹药库
    std::vector<cv::Point3f> object_points_flat;  // 传统的平面坐标 (无先验)
    std::vector<cv::Point3f> object_points_prior; // 带有 15度 扭转的立体坐标 (有先验)

public:
    Solver(const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs);
    bool solve(Armor &armor, bool use_prior = true);
};