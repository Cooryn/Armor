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
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double geometry_epsilon = 1e-6;
    static constexpr double process_noise = .01;
    static double wrap_to_pi(double angle);

    static bool valid_observation(const Eigen::Vector3d &z);
    static Eigen::Vector3d h(const Eigen::Matrix<double, 6, 1> &state);
    static Eigen::Matrix<double, 3, 6> jacobian(const Eigen::Matrix<double, 6, 1> &state);

    void initialize(const Eigen::Vector3d &z);
    void predict(double dt);
    bool update(const Eigen::Vector3d &z);

    PredictionResult update_frame(double timestamp_ms,
                                 const std::vector<Eigen::Vector4d> &observations, double horizon_ms = 50);

    Eigen::Matrix<double, 6, 1> X = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Identity() * 10;
    bool is_initialized = false;
private:
    double last_timestamp_ms = -1;
};
