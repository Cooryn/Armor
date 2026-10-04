#pragma once
#include <Eigen/Dense>
#include "predictor_armor.hpp"
#include "predictor_polar.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>

using SinglePlateState = Eigen::Matrix<double, 6, 1>; // [x, vx, y, vy, z, vz]
using SinglePlateCovariance = Eigen::Matrix<double, 6, 6>;
using SinglePlateObservation = Eigen::Vector3d; // [yaw rad, pitch rad, distance m]
using SinglePlateJacobian = Eigen::Matrix<double, 3, 6>;
constexpr double single_plate_pi = 3.14159265358979323846;
constexpr double single_plate_geometry_epsilon = 1e-6;
constexpr double single_plate_process_noise = .01;

struct SinglePlateEKF
{
    SinglePlateState state_ = SinglePlateState::Zero();
    SinglePlateCovariance covariance_ = SinglePlateCovariance::Identity() * 10;
    bool initialized_ = false;
};

double single_plate_wrap_to_pi(double angle);

bool single_plate_valid_observation(const SinglePlateObservation &z);
SinglePlateObservation single_plate_h(const SinglePlateState &s);
SinglePlateJacobian single_plate_jacobian(const SinglePlateState &s);
void single_plate_initialize(SinglePlateEKF &self, const SinglePlateObservation &z);
void single_plate_predict(SinglePlateEKF &self, double dt);
bool single_plate_update(SinglePlateEKF &self, const SinglePlateObservation &z);

enum PredictorType
{
    PREDICTOR_SINGLE_PLATE,
    PREDICTOR_POLAR,
    PREDICTOR_ARMOR
};

// Camera XYZ (m), then yaw/pitch/distance/plate yaw (rad/rad/m/rad).
struct VideoObservation
{
    Eigen::Vector3d position;
    Eigen::Vector4d measurement;
};
struct PredictionGeometry
{
    std::optional<Eigen::Vector3d> center;
    std::vector<Eigen::Vector4d> plates; // x, y, z, plate yaw
};
struct VideoPrediction
{
    bool initialized = false;
    PredictionGeometry current, future;
    std::string status = "waiting";
    double horizon_ms = 50;
    // Fixed numeric columns: SinglePlate=13, Polar=15, Armor=31 + status.
    // Initialization has no state row. Diagnostics may still be present.
    std::array<double, 31> values{};
    std::size_t value_count = 0;
    Eigen::Vector4d errors = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
    std::vector<Diagnostic> diagnostics;
};

struct VideoPredictor
{
    PredictorType type = PREDICTOR_ARMOR;
    double horizon_ms = 50;
    SinglePlateEKF basic;
    PolarEKF polar;
    ArmorEKF armor;
    std::int64_t last_frame = -1;
    double last_timestamp = -1;
};

VideoPrediction video_predictor_update(VideoPredictor &self, std::int64_t frame, double timestamp,
                                       const std::vector<VideoObservation> &observations);

struct CsvOutput
{
    std::ofstream stream_;
    bool first_ = true;
};
struct PredictionOutput
{
    PredictorType type;
    std::filesystem::path metrics_path;
    CsvOutput results, logs, futures;
    std::array<double, 4> squared_errors{};
    std::array<std::size_t, 4> error_counts{};
    bool finished = false;
};
PredictionOutput open_prediction_output(PredictorType model, const std::filesystem::path &directory,
                                        const std::string &suffix);
void prediction_output_write(PredictionOutput &self, std::int64_t frame, double timestamp,
                             const std::vector<VideoObservation> &obs, const VideoPrediction &prediction);
void prediction_output_finish(PredictionOutput &self);
