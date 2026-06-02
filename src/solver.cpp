#include "solver.hpp"

Solver::Solver(const cv::Mat &camera_mat, const cv::Mat &dist_coeffs)
{
    camera_matrix = camera_mat.clone();
    distort_coeffs = dist_coeffs.clone();

    const double HALF_X = 0.135 / 2.0;
    const double HALF_Y = 0.056 / 2.0;
    // 顺时针 3D 模型点 (TL, TR, BR, BL)
    object_points.push_back(cv::Point3f(-HALF_X, -HALF_Y, 0));
    object_points.push_back(cv::Point3f(HALF_X, -HALF_Y, 0));
    object_points.push_back(cv::Point3f(HALF_X, HALF_Y, 0));
    object_points.push_back(cv::Point3f(-HALF_X, HALF_Y, 0));
}

bool Solver::solve(Armor &armor)
{
    std::vector<cv::Point2f> image_points(armor.vertices, armor.vertices + 4);
    bool success = cv::solvePnP(object_points, image_points, camera_matrix, distort_coeffs,
                                armor.rvec, armor.tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (!success)
        return false;

    cv::Mat rotation_matrix;
    cv::Rodrigues(armor.rvec, rotation_matrix);

    cv::Mat projMatrix = cv::Mat::zeros(3, 4, CV_64F);
    rotation_matrix.copyTo(projMatrix(cv::Rect(0, 0, 3, 3)));

    cv::Mat camMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
    cv::Vec3d eulerAngles;
    cv::decomposeProjectionMatrix(projMatrix, camMatrix, rotMatrix, transVect,
                                  rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles);

    armor.yaw = eulerAngles[1];
    return true;
}

void Solver::drawAxis(cv::Mat &image, const Armor &armor)
{
    if (armor.tvec.empty() || armor.rvec.empty())
        return;
    float axis_length = 0.1f;
    std::vector<cv::Point3f> axis_points = {
        cv::Point3f(0, 0, 0), cv::Point3f(axis_length, 0, 0), cv::Point3f(0, axis_length, 0), cv::Point3f(0, 0, axis_length)};
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(axis_points, armor.rvec, armor.tvec, camera_matrix, distort_coeffs, projected_points);
    cv::line(image, projected_points[0], projected_points[1], cv::Scalar(0, 0, 255), 2);
    cv::line(image, projected_points[0], projected_points[2], cv::Scalar(0, 255, 0), 2);
    cv::line(image, projected_points[0], projected_points[3], cv::Scalar(255, 0, 0), 2);
}