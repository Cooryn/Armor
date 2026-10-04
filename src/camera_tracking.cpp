#include "camera_tracking.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "camera_frames.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

CameraGimbalOutput CameraGimbal::update(double timestamp_ms, const Eigen::Vector3d &center_camera,
                                        bool observed)
{
    for (double v : {config_.gain, config_.yaw_limit_deg, config_.pitch_limit_deg,
                     config_.yaw_speed_dps, config_.pitch_speed_dps,
                     config_.acceleration_dps2, config_.max_gap_ms})
        if (!std::isfinite(v) || v <= 0)
            throw std::invalid_argument("Controller limits and gain must be finite and positive");
    if (config_.yaw_limit_deg >= 90 || config_.pitch_limit_deg >= 90 ||
        !std::isfinite(config_.deadband_deg) || config_.deadband_deg < 0 ||
        config_.deadband_deg >= std::min(config_.yaw_limit_deg, config_.pitch_limit_deg))
        throw std::invalid_argument("Invalid angular limits or deadband");
    if (!std::isfinite(timestamp_ms) || timestamp_ms < 0 ||
        (last_time_ >= 0 && timestamp_ms <= last_time_))
        throw std::invalid_argument("Timestamps must be finite, nonnegative and strictly increasing");
    const double elapsed_ms = last_time_ >= 0 ? timestamp_ms - last_time_ : 0;
    const bool first = last_time_ < 0;
    last_time_ = timestamp_ms;
    CameraGimbalOutput out;
    // OpenCV camera coordinates: +x right, +y down, +z forward.
    out.target_valid = observed && center_camera.allFinite() && center_camera.z() > 1e-6;
    if (!out.target_valid)
    {
        last_rate_.setZero();
        out.status = observed ? "invalid_target" : "no_observation";
        return out;
    }
    constexpr double rad_to_deg = 180.0 / 3.14159265358979323846;
    out.yaw_error_deg = std::atan2(center_camera.x(), center_camera.z()) * rad_to_deg;
    out.pitch_error_deg = -std::atan2(center_camera.y(),
                                      std::hypot(center_camera.x(), center_camera.z())) *
                          rad_to_deg;
    out.yaw_offset_deg = std::clamp(out.yaw_error_deg, -config_.yaw_limit_deg, config_.yaw_limit_deg);
    out.pitch_offset_deg = std::clamp(out.pitch_error_deg, -config_.pitch_limit_deg, config_.pitch_limit_deg);
    out.angle_limited = out.yaw_offset_deg != out.yaw_error_deg ||
                        out.pitch_offset_deg != out.pitch_error_deg;
    if (first || elapsed_ms > config_.max_gap_ms)
    {
        last_rate_.setZero();
        out.status = first ? "initializing" : "time_gap";
        return out;
    }
    out.control_valid = true;
    Eigen::Vector2d desired(out.yaw_offset_deg, out.pitch_offset_deg);
    for (int i = 0; i < 2; ++i)
        desired(i) = config_.gain * std::copysign(
                                        std::max(0.0, std::abs(desired(i)) - config_.deadband_deg), desired(i));
    Eigen::Vector2d bounded(std::clamp(desired.x(), -config_.yaw_speed_dps, config_.yaw_speed_dps),
                            std::clamp(desired.y(), -config_.pitch_speed_dps, config_.pitch_speed_dps));
    out.speed_limited = (bounded - desired).squaredNorm() > 0;
    const double max_change = config_.acceleration_dps2 * elapsed_ms / 1000.0;
    Eigen::Vector2d rate = last_rate_ + (bounded - last_rate_).cwiseMax(-max_change).cwiseMin(max_change);
    out.acceleration_limited = (rate - bounded).squaredNorm() > 0;
    last_rate_ = rate;
    out.yaw_rate_dps = rate.x();
    out.pitch_rate_dps = rate.y();
    out.status = "tracking";
    return out;
}

void CameraTracking::update(const std::vector<Observation> &observations, double timestamp_ms)
{
    if (!std::isfinite(timestamp_ms) || timestamp_ms < 0 ||
        (last_time_ >= 0 && timestamp_ms <= last_time_))
        throw std::invalid_argument("Live frame timestamps must increase");
    const double dt = last_time_ >= 0 ? (timestamp_ms - last_time_) / 1000. : 0;
    accepted_ = 0;
    observations_ = observations;
    diagnostics_.clear();
    if (!ekf_.is_initialized)
    {
        const Observation *seed = nullptr;
        for (const auto &o : observations)
            if (ekf_.valid_observation(o.Z_obs) && (!seed || o.Z_obs(2) < seed->Z_obs(2)))
                seed = &o;
        if (seed)
        {
            ekf_.initialize(seed->Z_obs);
            accepted_ = 1;
            status_ = "initializing";
        }
        for (size_t i = 0; i < observations.size(); ++i)
        {
            Diagnostic d;
            d.observation_index = i;
            d.accepted = &observations[i] == seed;
            d.armor_id = d.accepted ? 0 : -1;
            d.reason = d.accepted ? "initialization" : (seed ? "initialization_unused" : "invalid");
            diagnostics_.push_back(d);
        }
    }
    else
    {
        ekf_.predict((timestamp_ms - last_time_) / 1000.);
        auto result = ekf_.update_multi(observations);
        accepted_ = result.first.size();
        diagnostics_ = std::move(result.second);
        status_ = accepted_ ? "tracking" : "prediction_only";
    }
    last_time_ = timestamp_ms;
    if (accepted_)
        last_observation_ = timestamp_ms;
    if (last_observation_ >= 0 && timestamp_ms - last_observation_ > 200)
        status_ = "lost";
    constexpr double radians = 3.14159265358979323846 / 180.;
    Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
    bool valid = true;
    if (mode_ == TRACKING_AUTOMATIC)
    {
        valid = accepted_ > 0 && ekf_.is_initialized;
        if (valid)
        {
            const auto future = ekf_.forecast(camera_tracking_horizon_s);
            int selected = -1;
            double distance = std::numeric_limits<double>::infinity();
            for (int i = 0; i < 4; ++i)
            {
                const Eigen::Vector3d p = future.plates.row(i).head(3).transpose();
                if (p.allFinite() && p.z() > 0 && p.squaredNorm() < distance)
                {
                    selected = i;
                    distance = p.squaredNorm();
                    predicted_target_ = p;
                }
            }
            valid = selected >= 0;
            const Eigen::Matrix3d rotation =
                (Eigen::AngleAxisd(simulated_angles_.x() * radians, Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(simulated_angles_.y() * radians, Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
            direction = rotation.transpose() * predicted_target_;
        }
    }
    else
    {
        const Eigen::Vector2d desired = mode_ == TRACKING_CENTER ? Eigen::Vector2d::Zero() : manual_target_;
        const Eigen::Vector2d error = (desired - simulated_angles_) * radians;
        direction = {std::sin(error.x()) * std::cos(error.y()), -std::sin(error.y()),
                     std::cos(error.x()) * std::cos(error.y())};
    }
    command_ = controller_.update(timestamp_ms, direction, valid);
    if (command_.control_valid)
    {
        const Eigen::Vector2d previous = simulated_angles_;
        simulated_angles_ += Eigen::Vector2d(command_.yaw_rate_dps, command_.pitch_rate_dps) * dt;
        simulated_angles_.x() = std::clamp(simulated_angles_.x(), -45., 45.);
        simulated_angles_.y() = std::clamp(simulated_angles_.y(), -30., 30.);
        if (mode_ != TRACKING_AUTOMATIC)
        {
            const Eigen::Vector2d desired = mode_ == TRACKING_CENTER ? Eigen::Vector2d::Zero() : manual_target_;
            for (int i = 0; i < 2; ++i)
                if ((desired(i) - previous(i)) * (desired(i) - simulated_angles_(i)) <= 0 ||
                    std::abs(desired(i) - simulated_angles_(i)) <= .001)
                    simulated_angles_(i) = desired(i);
        }
        if (dt > 0)
        {
            command_.yaw_rate_dps = (simulated_angles_.x() - previous.x()) / dt;
            command_.pitch_rate_dps = (simulated_angles_.y() - previous.y()) / dt;
        }
    }
}

void CameraTracking::key(int key)
{
    CameraTrackingMode mode = mode_;
    if (key == '1')
        mode = TRACKING_MANUAL;
    if (key == '2')
        mode = TRACKING_AUTOMATIC;
    if (key == '3' || key == 'c' || key == 'C')
        mode = TRACKING_CENTER;
    if (mode_ != mode)
    {
        mode_ = mode;
        manual_target_ = simulated_angles_;
        CameraGimbalConfig config;
        if (mode != TRACKING_AUTOMATIC)
            config.deadband_deg = 0;
        controller_ = CameraGimbal{config};
        command_ = {};
    }
    if (mode_ == TRACKING_MANUAL)
    {
        if (key == 'a' || key == 'A')
            manual_target_.x() -= 2;
        if (key == 'd' || key == 'D')
            manual_target_.x() += 2;
        if (key == 'w' || key == 'W')
            manual_target_.y() += 2;
        if (key == 's' || key == 'S')
            manual_target_.y() -= 2;
        manual_target_.x() = std::clamp(manual_target_.x(), -45., 45.);
        manual_target_.y() = std::clamp(manual_target_.y(), -30., 30.);
    }
}

const char *CameraTracking::mode_name() const
{
    return mode_ == TRACKING_MANUAL ? "MANUAL" : mode_ == TRACKING_AUTOMATIC ? "AUTO"
                                                                             : "CENTER";
}

bool CameraTracking::visible() const
{
    return ekf_.is_initialized && status_ != "lost";
}

static bool project(const std::vector<cv::Point3d> &points, const cv::Mat &camera,
                    const cv::Mat &distortion, std::vector<cv::Point> &pixels)
{
    for (const auto &p : points)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z <= .1)
            return false;
    std::vector<cv::Point2d> projected;
    cv::projectPoints(points, cv::Vec3d(), cv::Vec3d(), camera, distortion, projected);
    for (const auto &p : projected)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            return false;
        pixels.emplace_back(cvRound(std::clamp(p.x, -1e6, 1e6)), cvRound(std::clamp(p.y, -1e6, 1e6)));
    }
    return true;
}

void CameraTracking::draw(cv::Mat &image, const cv::Mat &camera, const cv::Mat &distortion, int frame_id)
{
    const int width = image.cols, height = image.rows;
    cv::Mat canvas(height, width + 340, CV_8UC3, cv::Scalar(30, 27, 24));
    image.copyTo(canvas(cv::Rect(0, 0, width, height)));
    cv::Mat view = canvas(cv::Rect(0, 0, width, height));
    const cv::Scalar colors[] = {{90, 100, 255}, {255, 180, 40}, {80, 230, 110}, {20, 190, 255}};
    const cv::Rect bounds(0, 0, width, height);
    std::vector<cv::Point> detail;
    std::optional<cv::Point> current_center;
    std::optional<cv::Point> plate_centers[4];
    auto pixel = [&](const cv::Point3d &point) -> std::optional<cv::Point>
    {
        std::vector<cv::Point> result;
        if (!project({point}, camera, distortion, result) || !bounds.contains(result[0]))
            return std::nullopt;
        detail.push_back(result[0]);
        return result[0];
    };
    auto plates = [&](const Forecast &forecast, bool future)
    {
        for (int id = 0; id < 4; ++id)
        {
            const auto &p = forecast.plates;
            const cv::Point3d center(p(id, 0), p(id, 1), p(id, 2));
            const cv::Point3d w(.0675 * std::cos(p(id, 3)), 0, .0675 * std::sin(p(id, 3))), h(0, .028, 0);
            std::vector<cv::Point> corners;
            if (project({center - w - h, center - w + h, center + w + h, center + w - h}, camera, distortion, corners))
            {
                for (int j = 0; j < 4; ++j)
                {
                    auto a = corners[j], b = corners[(j + 1) % 4];
                    if (!cv::clipLine(bounds, a, b))
                        continue;
                    detail.push_back(a);
                    detail.push_back(b);
                    if (future)
                    {
                        if (!cv::clipLine(view.size(), a, b))
                            continue;
                        const double length = cv::norm(b - a);
                        if (length < 1)
                            continue;
                        const cv::Point2d direction = cv::Point2d(b - a) / length;
                        for (double offset = 0; offset < length; offset += 10)
                            cv::line(view, cv::Point2d(a) + direction * offset,
                                     cv::Point2d(a) + direction * std::min(offset + 6, length), colors[id], 2, cv::LINE_AA);
                    }
                    else
                        cv::line(view, a, b, colors[id], 2, cv::LINE_AA);
                }
            }
            if (auto point = pixel(center))
            {
                if (future)
                    cv::drawMarker(view, *point, colors[id], cv::MARKER_DIAMOND, 11, 1, cv::LINE_AA);
                else
                {
                    plate_centers[id] = point;
                    cv::circle(view, *point, 8, colors[id], 2, cv::LINE_AA);
                    if (current_center)
                        cv::line(view, *current_center, *point, colors[id], 1, cv::LINE_AA);
                }
                cv::putText(view, std::string(future ? "F" : "A") + std::to_string(id),
                            *point + cv::Point(future ? 8 : 11, future ? 16 : -9), cv::FONT_HERSHEY_SIMPLEX,
                            future ? .45 : .5, colors[id], 1, cv::LINE_AA);
            }
        }
    };
    if (visible())
    {
        const auto current = ekf_.forecast(0), future = ekf_.forecast(camera_tracking_horizon_s);
        current_center = pixel({ekf_.X(0), ekf_.X(2), ekf_.X(4)});
        if (current_center)
        {
            if (drawn_time_ != last_time_)
            {
                trail_.push_back(*current_center);
                if (trail_.size() > 30)
                    trail_.pop_front();
            }
            for (size_t i = 1; i < trail_.size(); ++i)
                cv::line(view, trail_[i - 1], trail_[i], {180, 180, 180}, 1, cv::LINE_AA);
            cv::drawMarker(view, *current_center, {255, 255, 255}, cv::MARKER_STAR, 15, 2);
        }
        plates(current, false);
        if (auto point = pixel({future.state(0), future.state(2), future.state(4)}))
            cv::drawMarker(view, *point, {255, 255, 255}, cv::MARKER_DIAMOND, 13, 1);
        plates(future, true);
    }
    else
        trail_.clear();
    drawn_time_ = last_time_;
    constexpr double rad = 3.14159265358979323846 / 180.;
    const auto axis = (Eigen::AngleAxisd(simulated_angles_.x() * rad, Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(simulated_angles_.y() * rad, Eigen::Vector3d::UnitX())) *
                      Eigen::Vector3d::UnitZ();
    std::vector<cv::Point> optical_axis;
    if (project({{axis.x(), axis.y(), axis.z()}}, camera, distortion, optical_axis) && bounds.contains(optical_axis[0]))
    {
        cv::drawMarker(view, optical_axis[0], {255, 255, 0}, cv::MARKER_CROSS, 24, 2);
        cv::putText(view, "SIM CAMERA AXIS", optical_axis[0] + cv::Point(12, -12), cv::FONT_HERSHEY_SIMPLEX, .5, {255, 255, 0}, 1, cv::LINE_AA);
    }
    if (mode_ == TRACKING_AUTOMATIC && command_.target_valid)
    {
        std::vector<cv::Point> target;
        if (project({{predicted_target_.x(), predicted_target_.y(), predicted_target_.z()}}, camera, distortion, target) && bounds.contains(target[0]))
            cv::drawMarker(view, target[0], {255, 0, 255}, cv::MARKER_TILTED_CROSS, 20, 2);
    }
    size_t rejected = 0, unknown = 0;
    for (size_t i = 0; i < observations_.size(); ++i)
    {
        const auto &d = diagnostics_[i];
        const auto &z = observations_[i].Z_obs;
        const bool reject = d.reason == "invalid" || d.reason == "innovation_gate" || d.reason == "association_conflict";
        rejected += reject;
        unknown += !d.accepted && !reject;
        if (z.rows() != 4 || z.cols() != 1 || !z.allFinite() || z(2) <= 0)
            continue;
        const auto point = pixel({z(2) * std::cos(z(1)) * std::sin(z(0)), z(2) * std::sin(z(1)), z(2) * std::cos(z(1)) * std::cos(z(0))});
        if (!point)
            continue;
        const auto color = d.accepted ? colors[d.armor_id] : (reject ? cv::Scalar(40, 80, 255) : cv::Scalar(0, 220, 255));
        cv::drawMarker(view, *point, color, reject ? cv::MARKER_TILTED_CROSS : cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
        if (d.accepted && plate_centers[d.armor_id])
            cv::line(view, *point, *plate_centers[d.armor_id], color, 1, cv::LINE_AA);
    }
    const cv::Scalar white(235, 235, 235), gray(210, 210, 210);
    std::vector<std::pair<std::string, cv::Scalar>> lines = {{"ARMOR TRACKING", white},
                                                             {cv::format("Frame %d | %.2f s", frame_id, std::max(last_time_, 0.0) / 1000), gray},
                                                             {status_ == "tracking" ? "UPDATED" : status_ == "prediction_only" ? "PREDICTION ONLY"
                                                                                              : status_ == "lost"              ? "LOST"
                                                                                                                               : "INITIALIZING / NO STATE",
                                                              status_ == "tracking" ? cv::Scalar(80, 220, 100) : cv::Scalar(0, 200, 255)},
                                                             {"ESTIMATE (camera-frame EKF)", gray}};
    if (visible())
    {
        const auto &x = ekf_.X;
        lines.push_back({cv::format("xc: %.3f m", x(0)), white});
        lines.push_back({cv::format("yc: %.3f m", x(2)), white});
        lines.push_back({cv::format("zc: %.3f m", x(4)), white});
        lines.push_back({cv::format("body_yaw: %.3f rad", x(6)), white});
        lines.push_back({cv::format("w: %.2f rad/s", x(7)), white});
        lines.push_back({cv::format("r: %.3f m", x(8)), white});
        lines.push_back({cv::format("dl / dh: %.3f / %.3f", x(9), x(10)), white});
        lines.push_back({"FORECAST: +50 ms", white});
        lines.push_back({cv::format("Target time: %.3f s", std::max(last_time_, 0.0) / 1000 + camera_tracking_horizon_s), gray});
    }
    lines.insert(lines.end(), {{"", gray}, {"Observations: " + std::to_string(observations_.size()), white}, {"Accepted: " + std::to_string(accepted_) + "   Rejected: " + std::to_string(rejected), white}, {"Unclassified: " + std::to_string(unknown), {0, 220, 255}}, {"LEGEND", white}, {"Solid A0-A3: current estimate", gray}, {"Dashed F0-F3: future forecast", gray}, {"+ : observation", gray}, {"Red X: rejected observation", {40, 80, 255}}, {"Yellow +: not classified", {0, 220, 255}}, {"Star / trail: center", gray}, {"Diamond: future center/plate", gray}, {"A/F pairs: same relative ID", gray}, {"CAMERA TRACKING (preview)", white}});
    if (command_.target_valid)
        lines.push_back({cv::format("Yaw / pitch: %.2f / %.2f deg", command_.yaw_error_deg, command_.pitch_error_deg), gray});
    else
        lines.push_back({"Yaw / pitch: unavailable", gray});
    lines.push_back({cv::format("Rate: %.2f / %.2f deg/s", command_.yaw_rate_dps, command_.pitch_rate_dps), gray});
    lines.push_back({command_.control_valid ? "VALID | no motor commands" : "HOLD | no motor commands", {0, 220, 255}});
    lines.insert(lines.begin(), {{std::string("SIMULATION | ") + mode_name(), {0, 220, 255}},
                                 {"1 Manual | 2 Auto | 3/C Center", white},
                                 {"WASD: manual target +/-2 deg", gray},
                                 {cv::format("Sim yaw/pitch: %.1f / %.1f", simulated_angles_.x(), simulated_angles_.y()), white}});
    const int bottom = height >= 1000 ? height - 240 : height - 10;
    const int spacing = std::min(27, std::max(10, (bottom - 32) / int(lines.size())));
    for (size_t i = 0; i < lines.size(); ++i)
    {
        const int y = 32 + int(i) * spacing;
        if (y >= bottom)
            break;
        cv::putText(canvas, lines[i].first, {width + 16, y}, cv::FONT_HERSHEY_SIMPLEX, std::min(.53, spacing / 30.), lines[i].second, 1, cv::LINE_AA);
    }
    if (!detail.empty() && height >= 1000)
    {
        cv::Rect area = cv::boundingRect(detail);
        const cv::Point2d middle(area.x + area.width / 2., area.y + area.height / 2.);
        const int w = std::max(200, area.width + 70), h = std::max(140, area.height + 70);
        area = cv::Rect(int(middle.x - w / 2), int(middle.y - h / 2), w, h) & bounds;
        if (area.area() > 0)
        {
            const double scale = std::min(308. / area.width, 180. / area.height);
            cv::Mat zoom;
            cv::resize(view(area), zoom, {std::max(1, int(area.width * scale)), std::max(1, int(area.height * scale))});
            zoom.copyTo(canvas(cv::Rect(width + 16 + (308 - zoom.cols) / 2, height - 195, zoom.cols, zoom.rows)));
            cv::putText(canvas, "TARGET DETAIL", {width + 16, height - 210}, cv::FONT_HERSHEY_SIMPLEX, .5, gray, 1, cv::LINE_AA);
        }
    }
    image = canvas;
}

int export_control_csv(int argc, char **argv);

#ifndef CAMERA_TRACKING_LIBRARY
int main(int argc, char **argv)
{
    if (argc > 1 && std::string(argv[1]) == "--input")
        return export_control_csv(argc, argv);
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
    {
        std::cout << "Usage: " << argv[0] << " [red|blue] [video file] [--headless]\n"
                  << "Live preview at source FPS; Esc or close window to stop. --headless disables preview.\n"
                  << "Defaults: red video_1.avi. Keys: 1 manual, 2 auto, 3/C center; WASD manual steps.\n";
        std::cout << "Outputs: results/camera_prediction_*.avi and data/camera_pose_raw_*.csv\n";
        return 0;
    }
    if (argc > 4 || (argc == 4 && std::string(argv[3]) != "--headless"))
    {
        std::cerr << "Usage: " << argv[0] << " [red|blue] [video file] [--headless]" << std::endl;
        return 1;
    }
    const bool preview = argc < 4;
    std::string color = argc >= 2 ? argv[1] : "red";
    std::transform(color.begin(), color.end(), color.begin(),
                   [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    if (color != "red" && color != "blue")
    {
        std::cerr << "Color must be red or blue" << std::endl;
        return 1;
    }
    const EnemyColor target_color = color == "red" ? ENEMY_RED : ENEMY_BLUE;
    const std::string input_filename = argc >= 3 ? argv[2] : "video_1.avi";
    std::string input_path = input_filename;
    if (!std::filesystem::exists(input_path))
        input_path = (std::filesystem::path("assets/video") / input_filename).string();
    std::string extension = std::filesystem::path(input_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    if (extension != ".avi" && extension != ".mp4" && extension != ".mov" &&
        extension != ".mkv" && extension != ".m4v" && extension != ".webm" &&
        extension != ".mpg" && extension != ".mpeg" && extension != ".wmv")
    {
        std::cerr << "Unsupported video file extension: " << extension << std::endl;
        return 1;
    }
    cv::VideoCapture stream(input_path);
    cv::Mat original_frame;
    if (!stream.isOpened() || !stream.read(original_frame))
    {
        std::cerr << "Cannot read video: " << input_path << std::endl;
        return 1;
    }
    const double source_fps = stream.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(source_fps) || source_fps <= 0)
    {
        std::cerr << "Invalid source video FPS" << std::endl;
        return 1;
    }
    const char *window = "Camera prediction - Esc to stop";
    if (preview)
    {
        try
        {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            cv::resizeWindow(window, 960, 720);
        }
        catch (const cv::Exception &error)
        {
            std::cerr << "Cannot create OpenCV window; use --headless. " << error.what() << std::endl;
            return 1;
        }
    }
    constexpr int color_th = 70, gray_th = 170, min_area = 40;
    constexpr float min_angle = 55, max_angle_diff = 20;
    std::string stem = std::filesystem::path(input_path).stem().string();
    std::string out_dir = "./results/";
    std::filesystem::create_directories(out_dir);

    std::string output_path = out_dir + "camera_prediction_" + stem + ".avi";

    cv::VideoWriter writer;
    std::ofstream csv_file;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::Size output_size = original_frame.size();
    output_size.width += 340;
    writer.open(output_path, fourcc, source_fps, output_size, true);
    if (!writer.isOpened())
    {
        std::cerr << "Cannot open output video" << std::endl;
        return 1;
    }

    const auto separator = stem.find_last_of('_');
    std::string suffix = separator == std::string::npos ? "_" + stem : stem.substr(separator);
    std::string csv_filename = "camera_pose_raw" + suffix + ".csv";

    // 确保 ./data/ 目录存在
    std::filesystem::create_directories("./data/");
    csv_file.open("./data/" + csv_filename);
    if (!csv_file.is_open())
    {
        std::cerr << "Cannot open output CSV" << std::endl;
        return 1;
    }
    csv_file << std::setprecision(15);
    csv_file << "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,detection_score,reprojection_error,pnp_candidate_count,pnp_used_temporal,rvec_x,rvec_y,rvec_z,coordinate_frame\n";

    cv::Mat camera_matrix, distort_coeffs;

    if (stem.find("video_2") != std::string::npos)
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0.000000, 732.488057,
                         0.000000, 1714.616882, 546.930868,
                         0.000000, 0.000000, 1.000000);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.119922, -0.078593, 0.007511, -0.028028, 0.000000);
    }
    else
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                         0, 1288.1400736562441, 483.6163720308021,
                         0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
    }

    Solver pnp_solver{camera_matrix, distort_coeffs};

    CameraTracking tracking;
    std::ofstream controls(out_dir + "camera_control" + suffix + ".csv");
    if (!controls)
    {
        std::cerr << "Cannot open control CSV\n";
        return 1;
    }
    controls << std::setprecision(15);
    controls << "frame_id,timestamp_ms,mode,simulation_only,target_valid,yaw_deg,pitch_deg,yaw_rate_dps,pitch_rate_dps,control_valid,target_x,target_y,target_z\n";
    int frame_count = 0;

    // Frame timestamps use the source video clock.

    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        cv::Mat frame = original_frame.clone();

        cv::Mat mask = extractColor(frame, target_color, color_th, gray_th);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        std::vector<float> light_quality;
        auto lightRects = getValidLightRects(contours, (float)min_angle, &light_quality, 1.5, min_area);
        auto armors = matchArmors(lightRects, max_angle_diff, 2.f, .8f, .8f, 3.1f, .35f, light_quality);

        cv::Mat final_result = frame.clone();

        int text_y_offset = 30;
        std::vector<Armor> valid_armors;
        const double source_time = frame_count / source_fps;
        const auto yaw_hints = pnp_solver.yaw_hints(armors, source_time);

        for (size_t i = 0; i < armors.size(); i++)
        {
            bool solved = pnp_solver.solve(armors[i], yaw_hints[i]);
            if (solved)
            {
                valid_armors.push_back(armors[i]);

                double tx = armors[i].tvec.at<double>(0), ty = armors[i].tvec.at<double>(1), tz = armors[i].tvec.at<double>(2);
                double rx = armors[i].rvec.at<double>(0), ry = armors[i].rvec.at<double>(1), rz = armors[i].rvec.at<double>(2);

                cv::putText(final_result, cv::format("Armor[%zu] tvec: x %5.2f y %5.2f z %5.2f", i, tx, ty, tz),
                            cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 25;

                cv::putText(final_result, cv::format("Armor[%zu] rvec: x %5.2f y %5.2f z %5.2f", i, rx, ry, rz),
                            cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 30;
            }
        }

        pnp_solver.finish_frame(valid_armors, source_time);
        drawArmors(final_result, valid_armors);
        std::vector<Observation> observations;
        if (!valid_armors.empty())
        {
            // Source video time in milliseconds.
            double timestamp = frame_count * 1000.0 / source_fps;

            // 🚀 核心修改：不再只取 [0]，而是遍历所有合法的装甲板
            for (size_t i = 0; i < valid_armors.size(); i++)
            {
                Armor current_armor = valid_armors[i];

                // 1. 获取 PnP 算出的原始 x, y, z
                double tx = current_armor.tvec.at<double>(0);
                double ty = current_armor.tvec.at<double>(1);
                double tz = current_armor.tvec.at<double>(2);
                double armor_yaw_rad = current_armor.yaw * CV_PI / 180.0;

                // 2. 解算纯几何维度的目标朝向
                double target_yaw = std::atan2(tx, tz);
                double target_pitch = std::atan2(ty, std::sqrt(tx * tx + tz * tz));
                double distance = std::sqrt(tx * tx + ty * ty + tz * tz);

                // 将角度约束在 [-PI, PI]
                double armor_orientation_yaw = std::remainder(armor_yaw_rad, 2.0 * CV_PI);
                observations.push_back({Eigen::Vector4d(
                    target_yaw, target_pitch, distance, armor_orientation_yaw)});

                // 3. 写入 CSV (同一个 frame_count 会被写入多次，占多行)
                csv_file << frame_count << ","
                         << timestamp << ","
                         << tx << ","
                         << ty << ","
                         << tz << ","
                         << target_yaw << ","
                         << target_pitch << ","
                         << distance << ","
                         << armor_orientation_yaw << ","
                         << current_armor.detection_score << ","
                         << current_armor.reprojection_error << ","
                         << current_armor.pnp_candidate_count << ","
                         << current_armor.pnp_used_temporal << ","
                         << current_armor.rvec.at<double>(0) << ","
                         << current_armor.rvec.at<double>(1) << ","
                         << current_armor.rvec.at<double>(2) << ",camera\n";
            }
        }

        tracking.update(observations, frame_count * 1000.0 / source_fps);
        final_result = frame.clone();
        tracking.draw(final_result, camera_matrix, distort_coeffs, frame_count);

        const auto &command = tracking.command_;
        const auto &angles = tracking.simulated_angles_;
        const auto &target = tracking.predicted_target_;
        controls << frame_count << ',' << frame_count * 1000.0 / source_fps << ',' << tracking.mode_name()
                 << ",1," << (tracking.mode_ == TRACKING_AUTOMATIC && command.target_valid) << ',' << angles.x() << ',' << angles.y() << ',' << command.yaw_rate_dps << ','
                 << command.pitch_rate_dps << ',' << command.control_valid << ','
                 << target.x() << ',' << target.y() << ',' << target.z() << '\n';
        writer.write(final_result);
        ++frame_count;
        if (preview)
        {
            cv::imshow(window, final_result);
            const auto deadline = frame_start + std::chrono::duration<double>(1.0 / source_fps);
            bool stop = false;
            do
            {
                const double remaining_ms = std::chrono::duration<double, std::milli>(
                                                deadline - std::chrono::steady_clock::now())
                                                .count();
                const int delay = static_cast<int>(std::clamp(std::ceil(remaining_ms), 1.0, 10.0));
                const int key = cv::waitKey(delay) & 0xff;
                tracking.key(key);
                stop = key == 27;
                if (!stop)
                {
                    // Qt destroys its receiver when the last window is closed.
                    try
                    {
                        stop = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1;
                    }
                    catch (const cv::Exception &)
                    {
                        stop = true;
                    }
                }
                if (stop)
                    break;
            } while (std::chrono::steady_clock::now() < deadline);
            if (stop)
                break;
        }
        if (!stream.read(original_frame))
            break;

    } while (true);

    writer.release();
    csv_file.close();
    if (preview)
        cv::destroyAllWindows();

    std::cout << "视频已导出至 " << out_dir << " (共处理 " << frame_count << " 帧)" << std::endl;

    return 0;
}

#endif

static std::vector<std::vector<std::string>> read_csv(const std::filesystem::path &p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read " + p.string());
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool in_quote = false;
    char c;
    while (f.get(c))
    {
        if (c == '\"')
        {
            if (in_quote && f.peek() == '\"')
            {
                f.get(c);
                field += '\"';
            }
            else
                in_quote = !in_quote;
        }
        else if (!in_quote && (c == ',' || c == '\n' || c == '\r'))
        {
            row.push_back(field);
            field.clear();
            if (c != ',')
            {
                if (c == '\r' && f.peek() == '\n')
                    f.get(c);
                if (row.size() > 1 || !row[0].empty())
                    rows.push_back(row);
                row.clear();
            }
        }
        else
            field += c;
    }
    if (in_quote)
        throw std::invalid_argument("Unclosed CSV quote");
    if (!field.empty() || !row.empty())
    {
        row.push_back(field);
        rows.push_back(row);
    }
    if (rows.empty())
        throw std::invalid_argument("No columns to parse from file");
    if (rows[0][0].compare(0, 3, "\xef\xbb\xbf") == 0)
        rows[0][0].erase(0, 3);
    return rows;
}
static double numeric(const std::string &value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::numeric_limits<double>::quiet_NaN();
    const std::string text = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
    size_t used = 0;
    const double result = std::stod(text, &used);
    if (used != text.size())
        throw std::invalid_argument("Invalid numeric value: " + text);
    return result;
}
struct Record
{
    double frame, timestamp;
    CameraGimbalOutput command;
};
static void export_commands(const std::filesystem::path &input, const std::filesystem::path &output,
                            const CameraGimbalConfig &config, const std::optional<CameraFrames> &frames)
{
    CameraGimbal controller{config};
    const auto csv = read_csv(input);
    std::map<std::string, size_t> columns;
    for (size_t i = 0; i < csv.front().size(); ++i)
        if (!columns.emplace(csv.front()[i], i).second)
            throw std::invalid_argument("Duplicate CSV column");
    for (const char *key : {"frame_id", "timestamp", "xc", "yc", "zc", "accepted_count", "status"})
        if (!columns.count(key))
            throw std::invalid_argument(std::string("Missing CSV column: ") + key);
    std::vector<Record> records;
    double previous_frame = -1;
    std::string source_frame;
    for (size_t i = 1; i < csv.size(); ++i)
    {
        const auto &row = csv[i];
        if (row.size() != csv.front().size())
            throw std::invalid_argument("CSV row width mismatch at row " + std::to_string(i + 1));
        auto get = [&](const std::string &key)
        { return numeric(row.at(columns.at(key))); };
        const double frame = get("frame_id"), timestamp = get("timestamp"), accepted = get("accepted_count");
        if (!std::isfinite(frame) || frame < 0 || std::floor(frame) != frame || frame <= previous_frame)
            throw std::invalid_argument("Frame IDs must be increasing nonnegative integers");
        if (!std::isfinite(accepted) || accepted < 0 || std::floor(accepted) != accepted)
            throw std::invalid_argument("Invalid accepted_count");
        const auto &status = row.at(columns.at("status"));
        if ((status != "updated" && status != "prediction_only") ||
            (status == "updated") != (accepted > 0))
            throw std::invalid_argument("Inconsistent tracking status and accepted_count");
        const std::string frame_name = columns.count("coordinate_frame") ? row.at(columns.at("coordinate_frame")) : "camera";
        if ((frame_name != "camera" && frame_name != "base") ||
            (!source_frame.empty() && source_frame != frame_name))
            throw std::invalid_argument("Unknown or mixed input coordinate frames");
        source_frame = frame_name;
        if ((frame_name == "base") != frames.has_value())
            throw std::invalid_argument("Base-frame input requires telemetry/calibration; camera-frame input must omit them");
        Eigen::Vector3d center(get("xc"), get("yc"), get("zc"));
        if (frames)
        {
            const double reference = get("base_reference_timestamp_ms");
            const Eigen::Vector3d origin(get("base_origin_x_m"), get("base_origin_y_m"), get("base_origin_z_m"));
            if (!std::isfinite(reference) || !origin.allFinite() ||
                std::abs(reference - (*frames).reference_time_ms_) > 1e-6 ||
                (origin - (*frames).origin_in_mechanical_).norm() > 1e-9)
                throw std::invalid_argument("Base origin mismatch: use the same --origin-time-ms as pose_base");
            center = (*frames).at(timestamp).inverse() * center;
        }
        auto command = controller.update(timestamp, center,
                                         status == "updated");
        records.push_back({frame, timestamp, command});
        previous_frame = frame;
    }
    // Validate the whole input before replacing an existing command CSV.
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());
    std::ofstream out(output);
    if (!out)
        throw std::runtime_error("Cannot write output: " + output.string());
    out << "frame_id,timestamp,yaw_error_deg,pitch_error_deg,yaw_offset_deg,pitch_offset_deg,"
           "yaw_rate_dps,pitch_rate_dps,target_valid,control_valid,angle_limited,speed_limited,"
           "acceleration_limited,status\n"
        << std::setprecision(17) << std::boolalpha;
    for (const auto &r : records)
    {
        const auto &c = r.command;
        out << r.frame << ',' << r.timestamp << ',';
        if (std::isfinite(c.yaw_error_deg))
            out << c.yaw_error_deg;
        out << ',';
        if (std::isfinite(c.pitch_error_deg))
            out << c.pitch_error_deg;
        out << ',' << c.yaw_offset_deg << ',' << c.pitch_offset_deg << ','
            << c.yaw_rate_dps << ',' << c.pitch_rate_dps << ',' << c.target_valid << ','
            << c.control_valid << ',' << c.angle_limited << ',' << c.speed_limited << ','
            << c.acceleration_limited << ',' << c.status << '\n';
    }
    out.close();
    if (!out)
        throw std::runtime_error("Command CSV write failed");
    std::cout << "Saved " << records.size() << " camera control rows: " << output.string() << '\n';
}

int export_control_csv(int argc, char **argv)
{
    try
    {
        CameraGimbalConfig config;
        std::filesystem::path root = ".", input, output, telemetry, calibration;
        double sync_gap_ms = 100;
        double reference_time_ms = -1;
        std::string suffix = "1";
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            auto value = [&]()
            {
                if (++i >= argc)
                    throw std::invalid_argument("Missing value for " + argument);
                return std::string(argv[i]);
            };
            if (argument == "--suffix")
                suffix = value();
            else if (argument == "--root")
                root = std::filesystem::u8path(value());
            else if (argument == "--input")
                input = std::filesystem::u8path(value());
            else if (argument == "--output")
                output = std::filesystem::u8path(value());
            else if (argument == "--telemetry")
                telemetry = std::filesystem::u8path(value());
            else if (argument == "--calibration")
                calibration = std::filesystem::u8path(value());
            else if (argument == "--sync-gap-ms")
                sync_gap_ms = numeric(value());
            else if (argument == "--origin-time-ms")
            {
                reference_time_ms = numeric(value());
                if (!std::isfinite(reference_time_ms) || reference_time_ms < 0)
                    throw std::invalid_argument("Origin time must be finite and nonnegative");
            }
            else if (argument == "--gain")
                config.gain = numeric(value());
            else if (argument == "--yaw-limit-deg")
                config.yaw_limit_deg = numeric(value());
            else if (argument == "--pitch-limit-deg")
                config.pitch_limit_deg = numeric(value());
            else if (argument == "--yaw-speed-dps")
                config.yaw_speed_dps = numeric(value());
            else if (argument == "--pitch-speed-dps")
                config.pitch_speed_dps = numeric(value());
            else if (argument == "--acceleration-dps2")
                config.acceleration_dps2 = numeric(value());
            else if (argument == "--deadband-deg")
                config.deadband_deg = numeric(value());
            else if (argument == "--max-gap-ms")
                config.max_gap_ms = numeric(value());
            else if (argument == "--help" || argument == "-h")
            {
                std::cout << "Camera-only tracking CSV exporter. Angles: right-positive yaw, up-positive pitch.\n"
                             "--suffix N --root DIR --input CSV --output CSV\n"
                             "Base-frame input: --telemetry CSV --calibration CSV --sync-gap-ms 100 [--origin-time-ms MS]\n"
                             "--gain 2 --yaw-limit-deg 45 --pitch-limit-deg 30\n"
                             "--yaw-speed-dps 60 --pitch-speed-dps 45 --acceleration-dps2 180\n"
                             "--deadband-deg 0.2 --max-gap-ms 200\n"
                             "Offsets are relative to the current optical axis, not absolute encoder angles.\n";
                return 0;
            }
            else
                throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (input.empty())
            input = root / "results" / ("armor_prediction_result_" + suffix + ".csv");
        if (output.empty())
            output = root / "results" / ("camera_gimbal_" + suffix + ".csv");
        std::optional<CameraFrames> frames;
        if (telemetry.empty() != calibration.empty())
            throw std::invalid_argument("Specify both --telemetry and --calibration");
        if (reference_time_ms >= 0 && telemetry.empty())
            throw std::invalid_argument("--origin-time-ms requires telemetry/calibration");
        if (!telemetry.empty())
            frames = CameraFrames::load(telemetry, calibration, sync_gap_ms, reference_time_ms);
        for (const auto &source : {input, telemetry, calibration})
        {
            if (source.empty())
                continue;
            if (std::filesystem::weakly_canonical(source) == std::filesystem::weakly_canonical(output) ||
                (std::filesystem::exists(output) && std::filesystem::equivalent(source, output)))
                throw std::invalid_argument("Output must not overwrite any input file");
        }
        export_commands(input, output, config, frames);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}

// Video input and prediction presentation for the video applications.
const std::array<cv::Scalar, 4> plate_colors = {
    cv::Scalar(0, 255, 255), cv::Scalar(255, 180, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 100)};

static std::optional<cv::Point> project_prediction(const Eigen::Vector3d &p, const cv::Mat &matrix, const cv::Mat &distortion)
{
    if (!p.allFinite() || p(2) <= .1)
        return std::nullopt;
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(std::vector<cv::Point3d>{{p(0), p(1), p(2)}}, cv::Vec3d(0, 0, 0),
                      cv::Vec3d(0, 0, 0), matrix, distortion, pixels);
    const auto &pixel = pixels.front();
    constexpr double limit = double(std::numeric_limits<int>::max()) / 4;
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
        std::abs(pixel.x) >= limit || std::abs(pixel.y) >= limit)
        return std::nullopt;
    return cv::Point(cvRound(pixel.x), cvRound(pixel.y));
}

static void marker(cv::Mat &image, const Eigen::Vector3d &p, const cv::Mat &matrix, const cv::Mat &distortion,
                   const cv::Scalar &color, bool future)
{
    const auto pixel = project_prediction(p, matrix, distortion);
    if (!pixel || !cv::Rect(0, 0, image.cols, image.rows).contains(*pixel))
        return;
    if (future)
        cv::drawMarker(image, *pixel, color, cv::MARKER_DIAMOND, 14, 2, cv::LINE_AA);
    else
        cv::circle(image, *pixel, 5, color, 2, cv::LINE_AA);
}

static void draw_geometry(cv::Mat &image, const PredictionGeometry &geometry,
                          const cv::Mat &matrix, const cv::Mat &distortion, bool future, PredictorType type)
{
    if (geometry.center)
        marker(image, *geometry.center, matrix, distortion, cv::Scalar(255, 255, 255), future);
    for (std::size_t i = 0; i < geometry.plates.size(); ++i)
    {
        const auto &plate = geometry.plates[i];
        const Eigen::Vector3d center = plate.head<3>();
        const auto &color = plate_colors[i % plate_colors.size()];
        marker(image, center, matrix, distortion, color, future);
        if (type == PREDICTOR_SINGLE_PLATE)
            continue;
        const Eigen::Vector3d half_width(.0675 * std::cos(plate(3)), 0, .0675 * std::sin(plate(3))),
            half_height(0, .028, 0);
        const std::array<Eigen::Vector3d, 4> corners = {
            center - half_width - half_height, center + half_width - half_height,
            center + half_width + half_height, center - half_width + half_height};
        std::array<std::optional<cv::Point>, 4> pixels;
        for (int j = 0; j < 4; ++j)
            pixels[j] = project_prediction(corners[j], matrix, distortion);
        for (int j = 0; j < 4; ++j)
            if (pixels[j] && pixels[(j + 1) % 4])
            {
                auto a = *pixels[j], b = *pixels[(j + 1) % 4];
                if (!cv::clipLine(image.size(), a, b))
                    continue;
                if (!future)
                {
                    cv::line(image, a, b, color, 2, cv::LINE_AA);
                    continue;
                }
                const double length = cv::norm(b - a);
                if (length < 1)
                    continue;
                const cv::Point2d delta = cv::Point2d(b - a) / length;
                for (double start = 0; start < length; start += 12)
                    cv::line(image, cv::Point(cv::Point2d(a) + start * delta),
                             cv::Point(cv::Point2d(a) + std::min(start + 7, length) * delta), color, 2, cv::LINE_AA);
            }
    }
}

bool VideoInput::next()
{
    if (!stream_.read(image_))
    {
        if (std::isfinite(frame_count_) && double(frame_id_ + 1) < frame_count_)
            throw std::runtime_error("Video read failed before the final source frame");
        return false;
    }
    ++frame_id_;
    timestamp_ms = double(frame_id_) * 1000 / fps_;
    return true;
}

bool VideoOutput::write(const VideoInput &input, const Solver &solver, const SolvedFrame &poses, const VideoPrediction &prediction,
                        std::chrono::steady_clock::time_point frame_start)
{
    if (finished_)
        throw std::logic_error("Cannot write finished video output");
    for (std::size_t i = 0; i < poses.observations.size(); ++i)
    {
        const auto &a = poses.armors.at(i);
        const auto &p = poses.observations[i].position;
        const auto &z = poses.observations[i].measurement;
        raw_ << input.frame_id_ << ',' << input.timestamp_ms << ',' << p(0) << ',' << p(1) << ',' << p(2) << ','
             << z(0) << ',' << z(1) << ',' << z(2) << ',' << z(3) << ',' << a.detection_score << ','
             << a.reprojection_error << ',' << a.pnp_candidate_count << ',' << a.pnp_used_temporal << ','
             << a.rvec.at<double>(0) << ',' << a.rvec.at<double>(1) << ',' << a.rvec.at<double>(2) << ",camera\n";
    }
    predictions_.write(input.frame_id_, input.timestamp_ms, poses.observations, prediction);
    cv::Mat canvas = input.image_.clone();
    drawArmors(canvas, poses.armors);
    draw_geometry(canvas, prediction.current, solver.camera_matrix, solver.distort_coeffs, false, type_);
    draw_geometry(canvas, prediction.future, solver.camera_matrix, solver.distort_coeffs, true, type_);
    const char *model = type_ == PREDICTOR_SINGLE_PLATE ? "SinglePlate" : type_ == PREDICTOR_POLAR ? "Polar"
                                                                                                   : "Armor";
    cv::putText(canvas, cv::format("%s  frame %lld  %s  future +%.0f ms", model, static_cast<long long>(input.frame_id_), prediction.status.c_str(), prediction.horizon_ms),
                {20, 30}, cv::FONT_HERSHEY_SIMPLEX, .6, {0, 255, 255}, 2, cv::LINE_AA);
    for (std::size_t i = 0; i < poses.observations.size(); ++i)
    {
        const auto &p = poses.observations[i].position;
        cv::putText(canvas, cv::format("PnP[%zu] x %.2f y %.2f z %.2f m", i, p(0), p(1), p(2)),
                    {20, 55 + static_cast<int>(i) * 25}, cv::FONT_HERSHEY_SIMPLEX, .5, {0, 255, 255}, 1);
    }
    writer_.write(canvas);
    ++processed_;
    if (!preview_)
        return true;
    cv::imshow(window, canvas);
    const auto deadline = frame_start + std::chrono::duration<double>(1 / input.fps_);
    do
    {
        const double remaining = std::chrono::duration<double, std::milli>(
                                     deadline - std::chrono::steady_clock::now())
                                     .count();
        if ((cv::waitKey(static_cast<int>(std::clamp(std::ceil(remaining), 1.0, 10.0))) & 0xff) == 27)
            return false;
        // Closing the last Qt window can remove its receiver before this query.
        try
        {
            if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1)
                return false;
        }
        catch (const cv::Exception &)
        {
            return false;
        }
    } while (std::chrono::steady_clock::now() < deadline);
    return true;
}

void VideoOutput::finish()
{
    if (finished_)
        return;
    predictions_.finish();
    raw_.close();
    writer_.release();
    cv::VideoCapture exported(output_path_.string());
    if (!exported.isOpened() || exported.get(cv::CAP_PROP_FRAME_COUNT) != double(processed_))
        throw std::runtime_error("Output video write failed or frame count is incomplete: " + output_path_.string());
    if (preview_)
        cv::destroyAllWindows();
    finished_ = true;
    std::cout << "Processed " << processed_ << " frames. Video: " << output_path_.string()
              << "\nObservations: " << raw_path_.string() << "\nPredictions: " << output_path_.parent_path().string() << '\n';
}
