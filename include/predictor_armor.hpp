#pragma once
#include "output.hpp"
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
struct Match
{
    std::size_t index;
    int armor_id;
    double distance;
    Eigen::Vector4d residual;
};
struct Forecast
{
    Eigen::Matrix<double, 11, 1> state;
    Eigen::Matrix<double, 11, 11> covariance;
    Eigen::Matrix4d plates;
};

class predictor_armor
{
public:
    static constexpr double pi = 3.14159265358979323846;
    static double wrap_to_pi(double angle);
    static Eigen::Vector4d angular_residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted);
    static Eigen::Vector4d plate_pose(const Eigen::Matrix<double, 11, 1> &state, int armor_id);
    static Eigen::Vector4d h(const Eigen::Matrix<double, 11, 1> &state, int armor_id);
    static Eigen::Matrix<double, 4, 11> get_jacobian(const Eigen::Matrix<double, 11, 1> &state, int armor_id);

    PredictionResult update_frame(double timestamp_ms,
                                 const std::vector<Eigen::Vector4d> &observations, double horizon_ms = 50);
    void initialize(const Eigen::Vector4d &z);
    void predict(double dt);
    Forecast forecast(double horizon_s = .05) const;
    std::vector<Match> associate(const std::vector<Eigen::Vector4d> &observations) const;
    std::vector<Match> update_multi(const std::vector<Eigen::Vector4d> &observations);

    Eigen::Matrix<double, 11, 1> X = Eigen::Matrix<double, 11, 1>::Zero();
    Eigen::Matrix<double, 11, 11> P = (Eigen::Matrix<double, 11, 1>() << 10, 10, 10, 10, 10, 10, 10, 10, .01, .05, .05).finished().asDiagonal();
    Eigen::Matrix4d R = Eigen::Vector4d(.0016, .0016, .16, .0576).asDiagonal();
    double q_pos = 3, q_yaw = 15, q_r = 3e-4, q_dl = 3e-3, q_dh = 3e-3;
    bool is_initialized = false;
    double max_yaw_error = 45 * predictor_armor::pi / 180, max_distance_error = .5;

private:
    double last_timestamp_ms = -1;
    void transition(double dt, Eigen::Matrix<double, 11, 11> &transition_matrix, Eigen::Matrix<double, 11, 11> &Q) const;
};
