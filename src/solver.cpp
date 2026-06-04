#include "solver.hpp"

Solver::Solver(const cv::Mat &camera_matrix, const cv::Mat &distort_coeffs)
{
    this->camera_matrix = camera_matrix;
    this->distort_coeffs = distort_coeffs;

    const double LIGHT_HEIGHT = 0.056;
    const double LIGHT_WIDTH = 0.135;
    double half_x = LIGHT_WIDTH / 2.0;
    double half_y = LIGHT_HEIGHT / 2.0;

    // --- 1. 初始化传统的平面坐标 (无先验) ---
    object_points_flat.clear();
    object_points_flat.emplace_back(-half_x, -half_y, 0); // 左上
    object_points_flat.emplace_back(-half_x, half_y, 0);  // 左下
    object_points_flat.emplace_back(half_x, half_y, 0);   // 右下
    object_points_flat.emplace_back(half_x, -half_y, 0);  // 右上

    // --- 2. 初始化带有 15° 扭转的立体坐标 (有先验) ---
    double yaw_prior = 15.0 * CV_PI / 180.0;
    double cos_yaw = std::cos(yaw_prior);
    double sin_yaw = std::sin(yaw_prior);

    object_points_prior.clear();
    object_points_prior.emplace_back(-half_x * cos_yaw, -half_y, half_x * sin_yaw);
    object_points_prior.emplace_back(-half_x * cos_yaw, half_y, half_x * sin_yaw);
    object_points_prior.emplace_back(half_x * cos_yaw, half_y, -half_x * sin_yaw);
    object_points_prior.emplace_back(half_x * cos_yaw, -half_y, -half_x * sin_yaw);
}

bool Solver::solve(Armor &armor, bool use_prior)
{
    // 根据开关，选择对应的 3D 坐标模型
    const std::vector<cv::Point3f> &target_points = use_prior ? object_points_prior : object_points_flat;

    // ==========================================
    // 🚀 核心修复：把 C 风格数组打包成 OpenCV 喜欢的 std::vector
    // ==========================================
    std::vector<cv::Point2f> image_points(armor.vertices, armor.vertices + 4);

    // 传入打包好的 image_points，而不是原始的 armor.vertices
    bool success = cv::solvePnP(target_points, image_points, camera_matrix, distort_coeffs,
                                armor.rvec, armor.tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (!success)
        return false;

    // 将旋转向量转为旋转矩阵
    cv::Mat rot_mat;
    cv::Rodrigues(armor.rvec, rot_mat);

    // 提取 Yaw 角
    double raw_yaw = std::atan2(rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 2)) * 180.0 / CV_PI;

    // 补偿先验角度
    if (use_prior)
    {
        armor.yaw = raw_yaw + 15.0;
    }
    else
    {
        armor.yaw = raw_yaw;
    }

    return true;
}