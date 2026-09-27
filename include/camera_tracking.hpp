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

#include <Eigen/Geometry>
#include "predictor_armor.hpp"
#include <opencv2/core.hpp>
#include <deque>

// Live camera-frame tracking. Moving-platform compensation requires telemetry
// and must be performed before filtering; it is not inferred from the video.
class CameraTracking {
    predictor::ArmorEKF ekf_;
    CameraGimbal controller_;
public:
    enum class Mode { Manual, Automatic, Center };
private:
    Mode mode_ = Mode::Automatic;
    Eigen::Vector2d simulated_angles_ = Eigen::Vector2d::Zero();
    Eigen::Vector2d manual_target_ = Eigen::Vector2d::Zero();
    Eigen::Vector3d predicted_target_ = Eigen::Vector3d::Zero();
    void update_control(double timestamp_ms, double dt) {
        constexpr double radians = 3.14159265358979323846 / 180.;
        Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
        bool valid = true;
        if (mode_ == Mode::Automatic) {
            valid = accepted_ > 0 && ekf_.is_initialized;
            if (valid) {
                const auto future = ekf_.forecast(horizon_s);
                int selected = -1;
                double distance = std::numeric_limits<double>::infinity();
                for (int i = 0; i < 4; ++i) {
                    const Eigen::Vector3d p = future.plates.row(i).head(3).transpose();
                    if (p.allFinite() && p.z() > 0 && p.squaredNorm() < distance) {
                        selected = i;
                        distance = p.squaredNorm();
                        predicted_target_ = p;
                    }
                }
                valid = selected >= 0;
                const Eigen::Matrix3d rotation =
                    (Eigen::AngleAxisd(simulated_angles_.x()*radians, Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(simulated_angles_.y()*radians, Eigen::Vector3d::UnitX())).toRotationMatrix();
                direction = rotation.transpose() * predicted_target_;
            }
        } else {
            const Eigen::Vector2d desired = mode_ == Mode::Center ? Eigen::Vector2d::Zero() : manual_target_;
            const Eigen::Vector2d error = (desired - simulated_angles_) * radians;
            direction = {std::sin(error.x())*std::cos(error.y()), -std::sin(error.y()),
                         std::cos(error.x())*std::cos(error.y())};
        }
        command_ = controller_.update(timestamp_ms, direction, valid);
        if (command_.control_valid) {
            const Eigen::Vector2d previous = simulated_angles_;
            simulated_angles_ += Eigen::Vector2d(command_.yaw_rate_dps,command_.pitch_rate_dps)*dt;
            simulated_angles_.x() = std::clamp(simulated_angles_.x(),-45.,45.);
            simulated_angles_.y() = std::clamp(simulated_angles_.y(),-30.,30.);
            if (mode_ != Mode::Automatic) {
                const Eigen::Vector2d desired = mode_ == Mode::Center ? Eigen::Vector2d::Zero() : manual_target_;
                for (int i=0;i<2;++i)
                    if ((desired(i)-previous(i))*(desired(i)-simulated_angles_(i)) <= 0 ||
                        std::abs(desired(i)-simulated_angles_(i)) <= .001)
                        simulated_angles_(i)=desired(i);
            }
            if (dt > 0) {
                command_.yaw_rate_dps=(simulated_angles_.x()-previous.x())/dt;
                command_.pitch_rate_dps=(simulated_angles_.y()-previous.y())/dt;
            }
        }
    }
    std::optional<double> last_time_, last_observation_;
    size_t accepted_ = 0;
    std::string status_ = "waiting";
    CameraGimbal::Output command_;
    std::vector<predictor::Observation> observations_;
    std::vector<predictor::Diagnostic> diagnostics_;
    std::deque<cv::Point> trail_;
    std::optional<double> drawn_time_;
public:
    static constexpr double horizon_s = .05;
    void update(const std::vector<predictor::Observation> &observations, double timestamp_ms) {
        if (!std::isfinite(timestamp_ms) || timestamp_ms < 0 ||
            (last_time_ && timestamp_ms <= *last_time_))
            throw std::invalid_argument("Live frame timestamps must increase");
        const double dt = last_time_ ? (timestamp_ms - *last_time_) / 1000. : 0;
        accepted_ = 0;
        observations_ = observations;
        diagnostics_.clear();
        if (!ekf_.is_initialized) {
            const predictor::Observation *seed = nullptr;
            for (const auto &o : observations)
                if (ekf_.valid_observation(o.Z_obs) && (!seed || o.Z_obs(2) < seed->Z_obs(2)))
                    seed = &o;
            if (seed) {
                ekf_.initialize(seed->Z_obs);
                accepted_ = 1;
                status_ = "initializing";
            }
            for (size_t i=0; i<observations.size(); ++i) {
                predictor::Diagnostic d;
                d.observation_index = i;
                d.accepted = &observations[i] == seed;
                d.armor_id = d.accepted ? 0 : -1;
                d.reason = d.accepted ? "initialization" :
                    (seed ? "initialization_unused" : "invalid");
                diagnostics_.push_back(d);
            }
        } else {
            ekf_.predict((timestamp_ms - *last_time_) / 1000.);
            auto result = ekf_.update_multi(observations);
            accepted_ = result.first.size();
            diagnostics_ = std::move(result.second);
            status_ = accepted_ ? "tracking" : "prediction_only";
        }
        last_time_ = timestamp_ms;
        if (accepted_) last_observation_ = timestamp_ms;
        if (last_observation_ && timestamp_ms - *last_observation_ > 200) status_ = "lost";
        update_control(timestamp_ms, dt);
    }
    void set_mode(Mode mode) {
        if (mode_ == mode) return;
        mode_ = mode;
        manual_target_ = simulated_angles_;
        CameraGimbal::Config config;
        if (mode != Mode::Automatic) config.deadband_deg = 0;
        controller_ = CameraGimbal{config};
        command_ = {};
    }
    void key(int key) {
        if (key == '1') set_mode(Mode::Manual);
        if (key == '2') set_mode(Mode::Automatic);
        if (key == '3' || key == 'c' || key == 'C') set_mode(Mode::Center);
        if (mode_ == Mode::Manual) {
            if (key == 'a' || key == 'A') manual_target_.x() -= 2;
            if (key == 'd' || key == 'D') manual_target_.x() += 2;
            if (key == 'w' || key == 'W') manual_target_.y() += 2;
            if (key == 's' || key == 'S') manual_target_.y() -= 2;
            manual_target_.x() = std::clamp(manual_target_.x(),-45.,45.);
            manual_target_.y() = std::clamp(manual_target_.y(),-30.,30.);
        }
    }
    Mode mode() const { return mode_; }
    const char *mode_name() const {
        return mode_ == Mode::Manual ? "MANUAL" : mode_ == Mode::Automatic ? "AUTO" : "CENTER";
    }
    const Eigen::Vector2d &simulated_angles() const { return simulated_angles_; }
    const Eigen::Vector3d &predicted_target() const { return predicted_target_; }
    const predictor::ArmorEKF &filter() const { return ekf_; }
    const CameraGimbal::Output &command() const { return command_; }
    bool visible() const { return ekf_.is_initialized && status_ != "lost"; }
    const std::string &status() const { return status_; }
    size_t accepted() const { return accepted_; }
    void draw(cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion, int frame_id = 0);
};
