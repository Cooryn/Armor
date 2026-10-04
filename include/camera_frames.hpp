#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Fixed base: origin is the camera optical center at the reference time.
// Axes remain X right, Y down, Z forward at zero encoder angles.
// Camera: OpenCV optical axes, X right, Y down, Z forward.
// T_BC maps camera coordinates into the fixed base frame.
struct CameraCalibration
{
    Eigen::Vector3d yaw_to_pitch = Eigen::Vector3d::Zero();
    Eigen::Vector3d camera_in_pitch = Eigen::Vector3d::Zero();
    Eigen::Quaterniond camera_to_pitch = Eigen::Quaterniond::Identity();
};
struct CameraSample
{
    double timestamp_ms, yaw_deg, pitch_deg, roll_deg;
};
class CameraFrames
{
public:
    static CameraFrames load(const std::filesystem::path &telemetry,
                             const std::filesystem::path &calibration, double max_gap_ms = 100,
                             double reference_time_ms = -1);
    Eigen::Isometry3d at(double timestamp_ms) const;
    Eigen::Isometry3d reset_origin(double timestamp_ms);

    CameraCalibration calibration_;
    std::vector<CameraSample> samples_;
    double max_gap_ms_ = 100;
    Eigen::Vector3d origin_in_mechanical_ = Eigen::Vector3d::Zero();
    double reference_time_ms_ = 0;

private:
    Eigen::Isometry3d mechanical_at(double timestamp_ms) const;
    Eigen::Isometry3d transform(const CameraSample &s) const;
    static std::vector<std::map<std::string, double>> read_numeric_csv(const std::filesystem::path &path);
};

inline Eigen::Isometry3d CameraFrames::at(double timestamp_ms) const
{
    Eigen::Isometry3d pose = mechanical_at(timestamp_ms);
    pose.translation() -= origin_in_mechanical_;
    return pose;
}

inline Eigen::Isometry3d CameraFrames::reset_origin(double timestamp_ms)
{
    const Eigen::Vector3d new_origin = mechanical_at(timestamp_ms).translation();
    Eigen::Isometry3d new_from_old = Eigen::Isometry3d::Identity();
    new_from_old.translation() = origin_in_mechanical_ - new_origin;
    origin_in_mechanical_ = new_origin;
    reference_time_ms_ = timestamp_ms;
    return new_from_old;
}

inline CameraFrames CameraFrames::load(const std::filesystem::path &telemetry,
                                       const std::filesystem::path &calibration, double max_gap_ms,
                                       double reference_time_ms)
{
    const auto values = CameraFrames::read_numeric_csv(calibration);
    if (values.size() != 1)
        throw std::invalid_argument("Calibration CSV must contain exactly one row");
    const auto &r = values.front();
    CameraCalibration c;
    c.yaw_to_pitch = {r.at("axis_x_m"), r.at("axis_y_m"), r.at("axis_z_m")};
    c.camera_in_pitch = {r.at("camera_x_m"), r.at("camera_y_m"), r.at("camera_z_m")};
    c.camera_to_pitch = Eigen::Quaterniond(r.at("mount_qw"), r.at("mount_qx"),
                                           r.at("mount_qy"), r.at("mount_qz"));
    std::vector<CameraSample> samples;
    for (const auto &row : CameraFrames::read_numeric_csv(telemetry))
        samples.push_back({row.at("timestamp"), row.at("yaw_deg"), row.at("pitch_deg"), row.at("roll_deg")});
    CameraFrames self{std::move(c), std::move(samples), max_gap_ms};

    if (!self.calibration_.yaw_to_pitch.allFinite() || !self.calibration_.camera_in_pitch.allFinite() ||
        !self.calibration_.camera_to_pitch.coeffs().allFinite() ||
        std::abs(self.calibration_.camera_to_pitch.norm() - 1) > 1e-3 ||
        !std::isfinite(max_gap_ms) || max_gap_ms <= 0 || self.samples_.empty())
        throw std::invalid_argument("Invalid camera calibration, telemetry or synchronization limit");
    self.calibration_.camera_to_pitch.normalize();
    double previous = -1;
    for (const auto &s : self.samples_)
    {
        if (!std::isfinite(s.timestamp_ms) || s.timestamp_ms < 0 || s.timestamp_ms <= previous ||
            !std::isfinite(s.yaw_deg) || !std::isfinite(s.pitch_deg) || !std::isfinite(s.roll_deg))
            throw std::invalid_argument("Telemetry must have increasing finite timestamps and finite angles");
        previous = s.timestamp_ms;
    }
    self.reset_origin(reference_time_ms == -1 ? self.samples_.front().timestamp_ms : reference_time_ms);

    return self;
}

inline Eigen::Isometry3d CameraFrames::mechanical_at(double timestamp_ms) const
{
    if (!std::isfinite(timestamp_ms))
        throw std::invalid_argument("Invalid frame timestamp");
    auto right = std::lower_bound(samples_.begin(), samples_.end(), timestamp_ms,
                                  [](const CameraSample &s, double t)
                                  { return s.timestamp_ms < t; });
    // CSV serialization can round otherwise identical video/encoder times.
    constexpr double timestamp_tolerance_ms = 1e-6;
    if (right != samples_.end() && std::abs(right->timestamp_ms - timestamp_ms) <= timestamp_tolerance_ms)
        return transform(*right);
    if (right != samples_.begin() && std::abs(std::prev(right)->timestamp_ms - timestamp_ms) <= timestamp_tolerance_ms)
        return transform(*std::prev(right));
    if (right == samples_.begin() || right == samples_.end())
        throw std::invalid_argument("Frame timestamp is outside telemetry coverage (no extrapolation)");
    const auto &left = *std::prev(right);
    if (right->timestamp_ms - left.timestamp_ms > max_gap_ms_)
        throw std::invalid_argument("Telemetry synchronization gap exceeds limit");
    const double f = (timestamp_ms - left.timestamp_ms) / (right->timestamp_ms - left.timestamp_ms);
    auto angle = [f](double a, double b)
    { return a + f * std::remainder(b - a, 360.0); };
    return transform({timestamp_ms, angle(left.yaw_deg, right->yaw_deg),
                      angle(left.pitch_deg, right->pitch_deg), angle(left.roll_deg, right->roll_deg)});
}

inline Eigen::Isometry3d CameraFrames::transform(const CameraSample &s) const
{
    constexpr double radians = 3.14159265358979323846 / 180;
    const Eigen::Matrix3d yaw = Eigen::AngleAxisd(s.yaw_deg * radians, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d pitch_roll = (Eigen::AngleAxisd(s.pitch_deg * radians, Eigen::Vector3d::UnitX()) *
                                        Eigen::AngleAxisd(s.roll_deg * radians, Eigen::Vector3d::UnitZ()))
                                           .toRotationMatrix();
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = yaw * pitch_roll * calibration_.camera_to_pitch.toRotationMatrix();
    pose.translation() = yaw * (calibration_.yaw_to_pitch + pitch_roll * calibration_.camera_in_pitch);
    return pose;
}

inline std::vector<std::map<std::string, double>> CameraFrames::read_numeric_csv(const std::filesystem::path &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::invalid_argument("Cannot read frame configuration: " + path.string());
    std::string line;
    if (!std::getline(in, line))
        throw std::invalid_argument("Empty frame configuration");
    if (line.compare(0, 3, "\xef\xbb\xbf") == 0)
        line.erase(0, 3);
    auto split = [](const std::string &text)
    {
        std::vector<std::string> cells;
        std::stringstream stream(text);
        std::string cell;
        while (std::getline(stream, cell, ','))
        {
            const auto first = cell.find_first_not_of(" \t\r\n");
            cells.push_back(first == std::string::npos ? "" : cell.substr(first, cell.find_last_not_of(" \t\r\n") - first + 1));
        }
        if (!text.empty() && text.back() == ',')
            cells.emplace_back();
        return cells;
    };
    const auto headers = split(line);
    std::vector<std::map<std::string, double>> rows;
    while (std::getline(in, line))
    {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;
        const auto cells = split(line);
        if (cells.size() != headers.size())
            throw std::invalid_argument("Frame configuration CSV width mismatch");
        std::map<std::string, double> row;
        for (size_t i = 0; i < headers.size(); ++i)
        {
            size_t used;
            const double value = std::stod(cells[i], &used);
            if (used != cells[i].size() || !std::isfinite(value) || !row.emplace(headers[i], value).second)
                throw std::invalid_argument("Invalid frame configuration CSV");
        }
        rows.push_back(row);
    }
    return rows;
}
