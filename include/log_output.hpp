#pragma once
#include <Eigen/Dense>
#include <array>
#include <optional>
#include <limits>
#include <vector>
#include "camera.hpp"
#include "gimbal.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

enum PredictorType
{
    PREDICTOR_SINGLE_PLATE,
    PREDICTOR_ARMOR
};

struct PredictionGeometry
{
    std::optional<Eigen::Vector3d> center;
    std::vector<Eigen::Vector4d> plates;
};
struct PredictionResult
{
    PredictionGeometry current, future;
    std::string status = "waiting";
    // Model-specific log values; shared geometry, errors and frame metadata are written separately.
    // Single plate: prior/observed x, z, yaw, distance (8 values).
    // Armor: future_body_yaw, prior x/z, body_yaw, prior/observed armor_yaw, r, dl, dh (9 values).
    std::array<double, 9> values{};
    Eigen::Vector4d errors = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
};

class log_output
{
public:
    static std::filesystem::path result_directory(const std::filesystem::path &root,
                                                  const ::camera &camera, PredictorType model, bool base_frame);
    log_output() = default;
    log_output(PredictorType model, const std::filesystem::path &directory,
           const std::string &suffix, double horizon_ms, bool base_frame = false);
    void open_predictions(PredictorType model, const std::filesystem::path &directory,
                          const std::string &suffix, double horizon_ms, bool base_frame = false);
    void open(const std::filesystem::path &root, const ::camera &camera, PredictorType model,
              double horizon_ms, bool base_frame = true);
    void write(std::int64_t frame_id, double timestamp_ms, const PredictionResult &prediction);
    bool write(const ::camera &camera, const SolvedFrame &camera_poses,
               const PredictionResult &prediction, const ControlTarget &control,
               const SolvedFrame *base_poses = nullptr, const ::pose *pose_base = nullptr);
    void finish();

private:
    PredictorType type_ = PREDICTOR_ARMOR;
    double horizon_ms_ = 0;
    bool finished_ = true, predictions_finished_ = true;
    bool base_coordinates = false;
    std::filesystem::path metrics_path;
    std::ofstream raw_, base_, transforms_, controls_, results, futures;
    std::array<double, 4> squared_errors{};
    std::array<int, 4> error_counts{};

};
