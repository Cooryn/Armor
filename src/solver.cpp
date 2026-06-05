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

    // ==========================================
    // 纯粹的平面坐标 (Z 轴全为 0)
    // ==========================================
    object_points.clear();
    object_points.emplace_back(-half_x, -half_y, 0); // 左上
    object_points.emplace_back(-half_x, half_y, 0);  // 左下
    object_points.emplace_back(half_x, half_y, 0);   // 右下
    object_points.emplace_back(half_x, -half_y, 0);  // 右上
}

bool Solver::solve(Armor &armor)
{
    // 依然保留这层安全包装，防止 OpenCV 底层抛出 C 风格数组转换报错
    std::vector<cv::Point2f> image_points(armor.vertices, armor.vertices + 4);

    // 直接使用原始的 object_points 进行解算
    bool success = cv::solvePnP(object_points, image_points, camera_matrix, distort_coeffs,
                                armor.rvec, armor.tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (!success)
        return false;

    // 将旋转向量转为旋转矩阵
    cv::Mat rot_mat;
    cv::Rodrigues(armor.rvec, rot_mat);

    armor.yaw = std::atan2(rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 2)) * 180.0 / CV_PI;

    return true;
}