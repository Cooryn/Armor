#pragma once
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <cctype>
#include <algorithm>
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
    explicit CameraGimbal(const CameraGimbalConfig &config = CameraGimbalConfig{})
    {
        config_ = config;
    }
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
    explicit VideoInput(const std::filesystem::path &path)
    {
        path_ = path;

        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("Input video does not exist: " + path.string());
        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        constexpr std::array<const char *, 9> extensions = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"};
        if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
            throw std::invalid_argument("Unsupported video file extension: " + extension);
        stream_.open(path.string());
        if (!stream_.isOpened() || !stream_.read(image_))
            throw std::runtime_error("Cannot read video: " + path.string());
        fps_ = stream_.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps_) || fps_ <= 0)
            throw std::runtime_error("Invalid source video FPS");
        frame_count_ = stream_.get(cv::CAP_PROP_FRAME_COUNT);
        stem_ = path.stem().string();
        const auto separator = stem_.find_last_of('_');
        suffix_ = separator == std::string::npos ? stem_ : stem_.substr(separator + 1);
    }
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
                PredictorType model, bool preview)
    {
        predictions_ = PredictionOutput(model, root / "results", input.suffix_);
        type_ = model;
        preview_ = preview;
        output_path_ = root / "results" / (input.stem_ + ".avi");
        raw_path_ = root / "data" / ("pose_raw_" + input.suffix_ + ".csv");

        if (std::filesystem::weakly_canonical(input.path_) == std::filesystem::weakly_canonical(output_path_))
            throw std::invalid_argument("Output video must not overwrite the input video");
        writer_.open(output_path_.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), input.fps_, input.image_.size());
        if (!writer_.isOpened())
            throw std::runtime_error("Cannot write video: " + output_path_.string());
        std::filesystem::create_directories(root / "data");
        raw_.open(raw_path_);
        if (!raw_)
            throw std::runtime_error("Cannot write " + raw_path_.string());
        raw_.exceptions(std::ios::badbit | std::ios::failbit);
        raw_ << std::setprecision(15)
             << "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,detection_score,"
                "reprojection_error,pnp_candidate_count,pnp_used_temporal,rvec_x,rvec_y,rvec_z,coordinate_frame\n";
        if (preview_)
        {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            cv::resizeWindow(window, 960, 720);
        }
    }
    bool write(const VideoInput &input, const Solver &solver, const SolvedFrame &poses,
               const VideoPrediction &prediction, std::chrono::steady_clock::time_point frame_start);
    void finish();

private:
    static constexpr const char *window = "Armor - current / future prediction - Esc to stop";
    PredictorType type_;
    bool preview_, finished_ = false;
    std::filesystem::path output_path_, raw_path_;
    cv::VideoWriter writer_;
    std::ofstream raw_;
    PredictionOutput predictions_;
    std::int64_t processed_ = 0;
};
