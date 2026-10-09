#pragma once
#include "predictor.hpp"
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
    Eigen::Vector4d z;
};
struct Forecast
{
    Eigen::Matrix<double, 11, 1> state;
    Eigen::Matrix<double, 11, 11> covariance;
    Eigen::Matrix4d plates;
};
constexpr double armor_pi = 3.14159265358979323846;

class ArmorEKF
{
public:
    VideoPrediction update_frame(std::int64_t frame_id, double timestamp_ms,
                                 const std::vector<VideoObservation> &observations, double horizon_ms = 50);
    void initialize(const Eigen::Vector4d &z);
    void predict(double dt);
    Forecast forecast(double horizon_s = .05) const;
    std::vector<Match> associate(const std::vector<Eigen::Vector4d> &observations) const;
    std::vector<Match> update_multi(const std::vector<Eigen::Vector4d> &observations);

    Eigen::Matrix<double, 11, 1> X = Eigen::Matrix<double, 11, 1>::Zero();
    Eigen::Matrix<double, 11, 11> F = Eigen::Matrix<double, 11, 11>::Identity();
    Eigen::Matrix<double, 11, 11> P = (Eigen::Matrix<double, 11, 1>() << 10, 10, 10, 10, 10, 10, 10, 10, .01, .05, .05).finished().asDiagonal();
    Eigen::Matrix4d R = Eigen::Vector4d(.0016, .0016, .16, .0576).asDiagonal();
    double q_pos = 3, q_yaw = 15, q_r = 3e-4, q_dl = 3e-3, q_dh = 3e-3;
    bool is_initialized = false;
    double max_yaw_error = 45 * armor_pi / 180, max_distance_error = .5;

private:
    double last_timestamp_ms = -1;
    void transition(double dt, Eigen::Matrix<double, 11, 11> &transition_matrix, Eigen::Matrix<double, 11, 11> &Q) const;
};

double armor_wrap_to_pi(double angle);
Eigen::Vector4d armor_angular_residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted);
Eigen::Vector4d armor_plate_pose(const Eigen::Matrix<double, 11, 1> &state, int armor_id);
Eigen::Vector4d armor_h(const Eigen::Matrix<double, 11, 1> &state, int armor_id);
Eigen::Matrix<double, 4, 11> armor_get_jacobian(const Eigen::Matrix<double, 11, 1> &state, int armor_id);
