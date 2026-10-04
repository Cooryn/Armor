#pragma once
#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>
struct Observation
{
    Eigen::Vector4d Z_obs = Eigen::Vector4d::Zero();
    int armor_id = -1; // -1: automatic association; 0..3: specified plate.
};
struct Match
{
    std::size_t index;
    int armor_id;
    double nis;
    Eigen::Vector4d residual;
    Eigen::Matrix<double, 4, 11> H;
    Eigen::Vector4d Z_obs;
};
struct Diagnostic
{
    std::size_t observation_index;
    bool accepted = false;
    int armor_id = -1;
    double nis = std::numeric_limits<double>::quiet_NaN();
    std::string reason = "invalid";
    int best_candidate_id = -1;
    double distance_residual = std::numeric_limits<double>::quiet_NaN();
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
    void initialize(const Eigen::Vector4d &z);
    void predict(double dt);
    Forecast forecast(double horizon_s = .05) const;
    bool valid_observation(const Eigen::Vector4d &z) const;
    std::pair<std::vector<Match>, std::vector<Diagnostic>> associate(const std::vector<Observation> &observations) const;
    std::pair<std::vector<Match>, std::vector<Diagnostic>> update_multi(const std::vector<Observation> &observations);

    // Public matrices and parameters support calibration and offline evaluation.
    Eigen::Matrix<double, 11, 1> X = Eigen::Matrix<double, 11, 1>::Zero();
    Eigen::Matrix<double, 11, 11> F = Eigen::Matrix<double, 11, 11>::Identity();
    Eigen::Matrix<double, 11, 11> P = (Eigen::Matrix<double, 11, 1>() << 10, 10, 10, 10, 10, 10, 10, 10, .01, .05, .05).finished().asDiagonal();
    Eigen::Matrix4d R = Eigen::Vector4d(.0016, .0016, .16, .0576).asDiagonal();
    double q_pos = 3, q_yaw = 15, q_r = 3e-4, q_dl = 3e-3, q_dh = 3e-3;
    bool is_initialized = false, base_frame = false;
    double nis_gate = 16, pair_yaw_tolerance = 25 * armor_pi / 180, max_distance_error = .5;

private:
    std::pair<Eigen::Matrix<double, 11, 11>, Eigen::Matrix<double, 11, 11>> transition(double dt) const;
};

double armor_wrap_to_pi(double a);
Eigen::Vector4d armor_angular_residual(const Eigen::Vector4d &a, const Eigen::Vector4d &b);
Eigen::Vector4d armor_plate_pose(const Eigen::Matrix<double, 11, 1> &s, int id);
Eigen::Vector4d armor_h(const Eigen::Matrix<double, 11, 1> &s, int id);
Eigen::Matrix<double, 4, 11> armor_get_jacobian(const Eigen::Matrix<double, 11, 1> &s, int id);
