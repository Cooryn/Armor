#include "solver.hpp"
#include <cmath>

Solver::Solver(const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs)
{
    this->camera_matrix = camera_matrix;
    this->distort_coeffs = distort_coeffs;

    const double LIGHT_HEIGHT = 0.056;
    const double LIGHT_WIDTH = 0.135;
    double half_x = LIGHT_WIDTH / 2.0;
    double half_y = LIGHT_HEIGHT / 2.0;

    object_points.clear();
    object_points.emplace_back(-half_x, -half_y, 0); // 左上
    object_points.emplace_back(-half_x, half_y, 0);  // 左下
    object_points.emplace_back(half_x, half_y, 0);   // 右下
    object_points.emplace_back(half_x, -half_y, 0);  // 右上
}

bool Solver::solve(Armor &armor)
{
    std::vector<cv::Point2f> image_points(armor.vertices, armor.vertices + 4);

    bool success = cv::solvePnP(object_points, image_points, camera_matrix, distort_coeffs,
                                armor.rvec, armor.tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (!success)
        return false;

    // 将旋转向量转为旋转矩阵
    cv::Mat rot_mat;
    cv::Rodrigues(armor.rvec, rot_mat);
    rot_mat.convertTo(rot_mat, CV_64F);
    armor.yaw = std::atan2(rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 2)) * 180.0 / CV_PI;

    return true;
}