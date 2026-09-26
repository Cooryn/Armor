#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

// Visual tracking for a camera-only pan/tilt platform. Angles are offsets
// from the current optical axis, never absolute motor/encoder positions.
class CameraGimbal {
  public:
    struct Config {
        double gain = 2.0;              // angular velocity / angular error (1/s)
        double yaw_limit_deg = 45.0;    // maximum requested optical-axis offset
        double pitch_limit_deg = 30.0;
        double yaw_speed_dps = 60.0;
        double pitch_speed_dps = 45.0;
        double acceleration_dps2 = 180.0;
        double deadband_deg = 0.2;
        double max_gap_ms = 200.0;
    };
    struct Output {
        double yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
        double pitch_error_deg = std::numeric_limits<double>::quiet_NaN();
        double yaw_offset_deg = 0, pitch_offset_deg = 0;
        double yaw_rate_dps = 0, pitch_rate_dps = 0;
        bool target_valid = false, control_valid = false;
        bool angle_limited = false, speed_limited = false, acceleration_limited = false;
        std::string status = "hold";
    };

    explicit CameraGimbal(const Config &config) : config_(config) {
        for (double v : {config.gain, config.yaw_limit_deg, config.pitch_limit_deg,
                         config.yaw_speed_dps, config.pitch_speed_dps,
                         config.acceleration_dps2, config.max_gap_ms})
            if (!std::isfinite(v) || v <= 0)
                throw std::invalid_argument("Controller limits and gain must be finite and positive");
        if (config.yaw_limit_deg >= 90 || config.pitch_limit_deg >= 90 ||
            !std::isfinite(config.deadband_deg) || config.deadband_deg < 0 ||
            config.deadband_deg >= std::min(config.yaw_limit_deg, config.pitch_limit_deg))
            throw std::invalid_argument("Invalid angular limits or deadband");
    }
    CameraGimbal() : CameraGimbal(Config{}) {}

    Output update(double timestamp_ms, const Eigen::Vector3d &center_camera,
                  bool observed) {
        if (!std::isfinite(timestamp_ms) || timestamp_ms < 0 ||
            (last_time_ && timestamp_ms <= *last_time_))
            throw std::invalid_argument("Timestamps must be finite, nonnegative and strictly increasing");
        const double elapsed_ms = last_time_ ? timestamp_ms - *last_time_ : 0;
        const bool first = !last_time_;
        last_time_ = timestamp_ms;
        Output out;
        // OpenCV camera coordinates: +x right, +y down, +z forward.
        out.target_valid = observed && center_camera.allFinite() && center_camera.z() > 1e-6;
        if (!out.target_valid) {
            last_rate_.setZero();
            out.status = observed ? "invalid_target" : "no_observation";
            return out;
        }
        constexpr double rad_to_deg = 180.0 / 3.14159265358979323846;
        out.yaw_error_deg = std::atan2(center_camera.x(), center_camera.z()) * rad_to_deg;
        out.pitch_error_deg = -std::atan2(center_camera.y(),
                                  std::hypot(center_camera.x(), center_camera.z())) * rad_to_deg;
        out.yaw_offset_deg = std::clamp(out.yaw_error_deg, -config_.yaw_limit_deg, config_.yaw_limit_deg);
        out.pitch_offset_deg = std::clamp(out.pitch_error_deg, -config_.pitch_limit_deg, config_.pitch_limit_deg);
        out.angle_limited = out.yaw_offset_deg != out.yaw_error_deg ||
                            out.pitch_offset_deg != out.pitch_error_deg;
        if (first || elapsed_ms > config_.max_gap_ms) {
            last_rate_.setZero();
            out.status = first ? "initializing" : "time_gap";
            return out;
        }
        out.control_valid = true;
        Eigen::Vector2d desired(config_.gain * deadband(out.yaw_offset_deg),
                                config_.gain * deadband(out.pitch_offset_deg));
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

  private:
    double deadband(double angle) const {
        return std::copysign(std::max(0.0, std::abs(angle) - config_.deadband_deg), angle);
    }
    Config config_;
    std::optional<double> last_time_;
    Eigen::Vector2d last_rate_ = Eigen::Vector2d::Zero();
};
