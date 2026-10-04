#pragma once
#include <Eigen/Dense>
#include "predictor.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <deque>
#include <string>

// Camera-only pan/tilt control. Angles are optical-axis offsets.
struct CameraGimbalConfig
{
    double gain = 2.0;
    double yaw_limit_deg = 45.0, pitch_limit_deg = 30.0;
    double yaw_speed_dps = 60.0, pitch_speed_dps = 45.0;
    double acceleration_dps2 = 180.0;
    double deadband_deg = 0.2, max_gap_ms = 200.0;
};
struct CameraGimbalOutput
{
    double yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    double pitch_error_deg = std::numeric_limits<double>::quiet_NaN();
    double yaw_offset_deg = 0, pitch_offset_deg = 0;
    double yaw_rate_dps = 0, pitch_rate_dps = 0;
    bool target_valid = false, control_valid = false;
    bool angle_limited = false, speed_limited = false, acceleration_limited = false;
    std::string status = "hold";
};
class CameraGimbal
{
public:
    explicit CameraGimbal(const CameraGimbalConfig &config = CameraGimbalConfig{});
    CameraGimbalOutput update(double timestamp_ms, const Eigen::Vector3d &center_camera, bool observed);

    CameraGimbalConfig config_;

private:
    double last_time_ = -1; // Source time in ms; -1 means no previous frame.
    Eigen::Vector2d last_rate_ = Eigen::Vector2d::Zero();
};

enum CameraTrackingMode
{
    TRACKING_MANUAL,
    TRACKING_AUTOMATIC,
    TRACKING_CENTER
};
constexpr double camera_tracking_horizon_s = .05;
// Moving-platform compensation requires telemetry.
class CameraTracking
{
public:
    void update(const std::vector<Observation> &observations, double timestamp_ms);
    void key(int key);
    const char *mode_name() const;
    bool visible() const;
    void draw(cv::Mat &image, const cv::Mat &camera, const cv::Mat &distortion, int frame_id = 0);

    ArmorEKF ekf_ = ArmorEKF{};
    CameraGimbal controller_ = CameraGimbal{};
    CameraTrackingMode mode_ = TRACKING_AUTOMATIC;
    Eigen::Vector2d simulated_angles_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d manual_target_ = Eigen::Vector2d::Zero();
    Eigen::Vector3d predicted_target_ = Eigen::Vector3d::Zero();
    size_t accepted_ = 0;
    std::string status_ = "waiting";
    CameraGimbalOutput command_;
    std::vector<Diagnostic> diagnostics_;

private:
    double last_time_ = -1, last_observation_ = -1;
    std::vector<Observation> observations_;
    std::deque<cv::Point> trail_;
    double drawn_time_ = -1;
};
class VideoInput
{
public:
    explicit VideoInput(const std::filesystem::path &path);
    bool next();

    std::filesystem::path path_;
    cv::Mat image_;
    double fps_ = 0, timestamp_ms = 0;
    std::int64_t frame_id_ = 0;
    std::string stem_, suffix_;

private:
    cv::VideoCapture stream_;
    double frame_count_ = 0;
};
class Solver;
struct SolvedFrame;
class VideoOutput
{
public:
    VideoOutput(const std::filesystem::path &root, const VideoInput &input,
                PredictorType model, bool preview);
    bool write(const VideoInput &input, const Solver &solver, const SolvedFrame &poses,
               const VideoPrediction &prediction, std::chrono::steady_clock::time_point frame_start);
    void finish();

private:
    PredictorType type_;
    bool preview_, finished_ = false;
    std::filesystem::path output_path_, raw_path_;
    cv::VideoWriter writer_;
    std::ofstream raw_;
    PredictionOutput predictions_;
    std::int64_t processed_ = 0;
};
