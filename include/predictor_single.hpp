#pragma once
#include "output.hpp"
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <optional>
#include <limits>
#include <string>
#include <vector>

class predictor_single
{
public:
    static const double pi;
    static double wrap_to_pi(double angle);
    static Eigen::Vector4d angular_residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted);

    static Eigen::Vector4d h(const Eigen::Matrix<double, 8, 1> &state);
    static Eigen::Matrix<double, 4, 8> jacobian(const Eigen::Matrix<double, 8, 1> &state);

    void initialize(const Eigen::Vector4d &z);
    void predict(double dt);
    bool update(const Eigen::Vector4d &z);

    PredictionResult update_frame(double timestamp_ms,
                                  const std::vector<Eigen::Vector4d> &observations, double horizon_ms = 50);

    Eigen::Matrix<double, 8, 1> X = Eigen::Matrix<double, 8, 1>::Zero();
    Eigen::Matrix<double, 8, 8> P = Eigen::Matrix<double, 8, 8>::Identity() * 10;
    Eigen::Matrix4d R = Eigen::Vector4d(.000025, .000025, .01, .0576).asDiagonal();
    double q_pos = 100, q_yaw = 100;
    double geometry_epsilon = 1e-6;
    bool is_initialized = false;

private:
    double last_timestamp_ms = -1;
};
