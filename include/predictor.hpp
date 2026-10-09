#pragma once
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <optional>
#include <limits>
#include <string>
#include <vector>

constexpr double single_plate_pi = 3.14159265358979323846;
constexpr double single_plate_geometry_epsilon = 1e-6;
constexpr double single_plate_process_noise = .01;

enum PredictorType
{
    PREDICTOR_SINGLE_PLATE,
    PREDICTOR_POLAR,
    PREDICTOR_ARMOR
};

struct VideoObservation
{
    Eigen::Vector3d position;
    Eigen::Vector4d measurement;
};
struct PredictionGeometry
{
    std::optional<Eigen::Vector3d> center;
    std::vector<Eigen::Vector4d> plates;
};
struct VideoPrediction
{
    bool initialized = false;
    PredictionGeometry current, future;
    std::string status = "waiting";
    double horizon_ms = 50;
    double position_variance = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 31> values{};
    std::size_t value_count = 0;
    Eigen::Vector4d errors = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
};

class SinglePlateEKF
{
public:
    void initialize(const Eigen::Vector3d &z);
    void predict(double dt);
    bool update(const Eigen::Vector3d &z);

    VideoPrediction update_frame(std::int64_t frame, double timestamp_ms,
                                 const std::vector<VideoObservation> &observations, double horizon_ms = 50);

    Eigen::Matrix<double, 6, 1> state_ = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity() * 10;
    bool initialized_ = false;
private:
    double last_timestamp_ms_ = -1;
};

double single_plate_wrap_to_pi(double angle);

bool single_plate_valid_observation(const Eigen::Vector3d &z);
Eigen::Vector3d single_plate_h(const Eigen::Matrix<double, 6, 1> &s);
Eigen::Matrix<double, 3, 6> single_plate_jacobian(const Eigen::Matrix<double, 6, 1> &s);
