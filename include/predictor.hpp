#pragma once
#include <stdexcept>
#include <iomanip>
#include <Eigen/Dense>
#include "predictor_armor.hpp"
#include "predictor_polar.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>

constexpr double single_plate_pi = 3.14159265358979323846;
constexpr double single_plate_geometry_epsilon = 1e-6;
constexpr double single_plate_process_noise = .01;

class SinglePlateEKF
{
public:
    // Observation: [yaw rad, pitch rad, distance m].
    void initialize(const Eigen::Vector3d &z);
    void predict(double dt);
    bool update(const Eigen::Vector3d &z);

    // State: [x, vx, y, vy, z, vz].
    Eigen::Matrix<double, 6, 1> state_ = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity() * 10;
    bool initialized_ = false;
};

double single_plate_wrap_to_pi(double angle);

bool single_plate_valid_observation(const Eigen::Vector3d &z);
Eigen::Vector3d single_plate_h(const Eigen::Matrix<double, 6, 1> &s);
Eigen::Matrix<double, 3, 6> single_plate_jacobian(const Eigen::Matrix<double, 6, 1> &s);

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

class VideoPredictor
{
public:
    VideoPredictor(PredictorType model = PREDICTOR_ARMOR, double prediction_horizon_ms = 50)
    {
        type = model;
        horizon_ms = prediction_horizon_ms;
    }
    VideoPrediction update(std::int64_t frame, double timestamp,
                           const std::vector<VideoObservation> &observations);

private:
    PredictorType type = PREDICTOR_ARMOR;
    double horizon_ms = 50;
    SinglePlateEKF basic;
    PolarEKF polar;
    ArmorEKF armor;
    std::int64_t last_frame = -1;
    double last_timestamp = -1;
};

struct CsvOutput
{
    std::ofstream stream_;
    bool first_ = true;
};
class PredictionOutput
{
public:
    PredictionOutput()
    {
        type = PREDICTOR_ARMOR;
        finished = true;
    }
    PredictionOutput(PredictorType model, const std::filesystem::path &directory,
                     const std::string &suffix)
    {
        constexpr const char *basic_header =
            "frame_id,predicted_x,observed_x,error_x,predicted_z,observed_z,error_z,"
            "predicted_yaw,observed_yaw,error_yaw,predicted_distance,observed_distance,error_distance";
        constexpr const char *polar_header =
            "frame_id,xc,vxc,yc,vyc,zc,vzc,body_yaw,w,r,err_target_yaw,err_target_pitch,"
            "err_distance,err_armor_yaw,obs_armor_yaw";
        constexpr const char *armor_header =
            "frame_id,timestamp,prediction_horizon_ms,prediction_timestamp,future_xc,future_yc,future_zc,"
            "future_body_yaw,xc,yc,zc,vxc,vyc,vzc,w,xa,za,armor_id,body_yaw,pred_armor_yaw,obs_armor_yaw,"
            "err_target_yaw,err_target_pitch,err_distance,err_armor_yaw,r,dl,dh,observation_count,"
            "accepted_count,rejected_count,status";

        type = model;

        if (suffix.empty() || suffix.find_first_of("/\\") != std::string::npos)
            throw std::invalid_argument("Invalid output suffix");
        const char *prefix;
        const char *header;
        switch (type)
        {
        case PREDICTOR_SINGLE_PLATE:
            prefix = "";
            header = basic_header;
            break;
        case PREDICTOR_POLAR:
            prefix = "polar_";
            header = polar_header;
            break;
        case PREDICTOR_ARMOR:
            prefix = "armor_";
            header = armor_header;
            break;
        default:
            throw std::invalid_argument("Unknown predictor type");
        }
        auto open = [](CsvOutput &csv, const std::filesystem::path &path, const char *header)
        {
            csv.stream_.open(path);
            if (!csv.stream_)
                throw std::runtime_error("Cannot write " + path.string());
            csv.stream_.exceptions(std::ios::badbit | std::ios::failbit);
            csv.stream_ << header << '\n'
                        << std::setprecision(17);
        };
        std::filesystem::create_directories(directory);
        open(results, directory / (std::string(prefix) + "prediction_result_" + suffix + ".csv"), header);
        metrics_path = directory / (std::string(prefix) + "rmse_result_" + suffix + ".txt");
        if (type == PREDICTOR_ARMOR)
        {
            open(logs, directory / ("armor_observation_diagnostics_" + suffix + ".csv"),
                 "frame_id,observation_index,accepted,armor_id,nis,reason,timestamp,observed_distance,"
                 "observed_armor_yaw,best_candidate_id,distance_residual");
            open(futures, directory / ("armor_future_prediction_" + suffix + ".csv"),
                 "frame_id,timestamp,prediction_timestamp,prediction_horizon_ms,armor_id,x,y,z,"
                 "armor_orientation_yaw,source_status");
        }
    }
    void write(std::int64_t frame, double timestamp,
               const std::vector<VideoObservation> &observations, const VideoPrediction &prediction);
    void finish();

private:
    PredictorType type;
    std::filesystem::path metrics_path;
    CsvOutput results, logs, futures;
    std::array<double, 4> squared_errors{};
    std::array<std::size_t, 4> error_counts{};
    bool finished = false;
};
