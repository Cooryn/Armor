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

class output
{
public:
    static std::filesystem::path result_directory(const std::filesystem::path &root,
                                                  const ::camera &camera, PredictorType model, bool base_frame);
    output() = default;
    output(PredictorType model, const std::filesystem::path &directory,
           const std::string &suffix, double horizon_ms, bool base_frame = false);
    void open_predictions(PredictorType model, const std::filesystem::path &directory,
                          const std::string &suffix, double horizon_ms, bool base_frame = false);
    void open(const std::filesystem::path &root, const ::camera &camera, PredictorType model,
              bool preview, double horizon_ms, bool base_frame = true);
    void write(std::int64_t frame_id, double timestamp_ms, const PredictionResult &prediction);
    bool write(const ::camera &camera, const ::solver &solver, const SolvedFrame &camera_poses,
               const PredictionResult &prediction, const ControlTarget &control,
               std::chrono::steady_clock::time_point frame_start,
               const SolvedFrame *base_poses = nullptr, const ::pose *pose_base = nullptr);
    void finish();

private:
    bool write_canvas(const ::camera &camera, const cv::Mat &canvas,
                      std::chrono::steady_clock::time_point frame_start);
    static void draw_prediction(cv::Mat &image, const PredictionGeometry &geometry,
                                const Eigen::Isometry3d &camera_from_base, const ::solver &solver,
                                bool future);
    static const char *const window;
    PredictorType type_ = PREDICTOR_ARMOR;
    double horizon_ms_ = 0;
    bool preview_ = false, finished_ = true, predictions_finished_ = true;
    bool base_coordinates = false;
    std::filesystem::path output_path_, metrics_path;
    cv::VideoWriter writer_;
    std::ofstream raw_, base_, transforms_, controls_, results, futures;
    std::array<double, 4> squared_errors{};
    std::array<std::size_t, 4> error_counts{};
    std::int64_t processed_ = 0, video_frames_ = 0;
    double video_fps_ = 0, first_timestamp_ms_ = 0, last_timestamp_ms_ = 0;
    double measured_start_ms_ = 0, read_total_ms_ = 0;
    cv::Mat previous_canvas_;
};
