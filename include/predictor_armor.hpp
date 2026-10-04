#pragma once
#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>
using ArmorState = Eigen::Matrix<double, 11, 1>;
using ArmorCovariance = Eigen::Matrix<double, 11, 11>;
using ArmorJacobian = Eigen::Matrix<double, 4, 11>;
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
    ArmorJacobian H;
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
    ArmorState state;
    ArmorCovariance covariance;
    Eigen::Matrix4d plates;
};
using Association = std::pair<std::vector<Match>, std::vector<Diagnostic>>;
using ArmorMeasurement = Eigen::Vector4d;
using ArmorJointJacobian = Eigen::Matrix<double, Eigen::Dynamic, 11, 0, 16, 11>;
using ArmorJointResidual = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 16, 1>;
using ArmorJointNoise = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16>;
using ArmorGain = Eigen::Matrix<double, 11, Eigen::Dynamic, 0, 11, 16>;
constexpr double armor_pi = 3.14159265358979323846;

// State only. Association and joint updates remain unchanged.
struct ArmorEKF
{
    ArmorState X = ArmorState::Zero();
    ArmorCovariance F = ArmorCovariance::Identity();
    ArmorCovariance P = (ArmorState() << 10, 10, 10, 10, 10, 10, 10, 10, .01, .05, .05).finished().asDiagonal();
    Eigen::Matrix4d R = ArmorMeasurement(.0016, .0016, .16, .0576).asDiagonal();
    double q_pos = 3, q_yaw = 15, q_r = 3e-4, q_dl = 3e-3, q_dh = 3e-3;
    bool is_initialized = false, base_frame = false;
    double nis_gate = 16, pair_yaw_tolerance = 25 * armor_pi / 180, max_distance_error = .5;
};

double armor_wrap_to_pi(double a);
ArmorMeasurement armor_angular_residual(const ArmorMeasurement &a, const ArmorMeasurement &b);
std::pair<ArmorCovariance, ArmorCovariance> armor_transition(const ArmorEKF &self, double dt);
void armor_predict(ArmorEKF &self, double dt);
Forecast armor_forecast(const ArmorEKF &self, double horizon_s = .05);
ArmorMeasurement armor_plate_pose(const ArmorState &s, int id);
ArmorMeasurement armor_h(const ArmorState &s, int id);
ArmorJacobian armor_get_jacobian(const ArmorState &s, int id);
bool armor_valid_observation(const ArmorEKF &self, const ArmorMeasurement &z);
void armor_initialize(ArmorEKF &self, const ArmorMeasurement &z);

Association armor_associate(const ArmorEKF &self, const std::vector<Observation> &observations);
Association armor_update_multi(ArmorEKF &self, const std::vector<Observation> &obs);
