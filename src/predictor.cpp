#include "predictor.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <initializer_list>
#include <stdexcept>

static bool single_plate_valid_geometry(const SinglePlateState &s);

double single_plate_wrap_to_pi(double angle) {
    double result = std::fmod(angle + single_plate_pi, 2 * single_plate_pi);
    return (result < 0 ? result + 2 * single_plate_pi : result) - single_plate_pi;
}

bool single_plate_valid_observation(const SinglePlateObservation &z) {
    return z.allFinite() && z(2) > single_plate_geometry_epsilon && std::abs(z(1)) < single_plate_pi / 2 &&
           z(2) * std::cos(z(1)) > single_plate_geometry_epsilon;
}

SinglePlateObservation single_plate_h(const SinglePlateState &s) {
    const double horizontal = std::hypot(s(0), s(4));
    return {std::atan2(s(0), s(4)), std::atan2(s(2), horizontal), std::hypot(horizontal, s(2))};
}

SinglePlateJacobian single_plate_jacobian(const SinglePlateState &s) {
    if (!single_plate_valid_geometry(s))
        throw std::invalid_argument("Singular single-plate observation geometry");
    const double x = s(0), y = s(2), z = s(4), horizontal = std::hypot(x, z),
                 distance = std::hypot(horizontal, y), horizontal2 = horizontal * horizontal,
                 distance2 = distance * distance;
    SinglePlateJacobian H = SinglePlateJacobian::Zero();
    H(0, 0) = z / horizontal2;
    H(0, 4) = -x / horizontal2;
    H(1, 0) = -x * y / (horizontal * distance2);
    H(1, 2) = horizontal / distance2;
    H(1, 4) = -y * z / (horizontal * distance2);
    H(2, 0) = x / distance;
    H(2, 2) = y / distance;
    H(2, 4) = z / distance;
    return H;
}

void single_plate_initialize(SinglePlateEKF &self, const SinglePlateObservation &z) {
    if (!single_plate_valid_observation(z))
        throw std::invalid_argument("Invalid single-plate initialization observation");
    const double horizontal = z(2) * std::cos(z(1));
    self.state_ << horizontal * std::sin(z(0)), 0, z(2) * std::sin(z(1)), 0,
        horizontal * std::cos(z(0)), 0;
    self.covariance_ = SinglePlateCovariance::Identity() * 10;
    self.initialized_ = true;
}

void single_plate_predict(SinglePlateEKF &self, double dt) {
    if (!self.initialized_ || !std::isfinite(dt) || dt < 0)
        throw std::invalid_argument("Prediction requires initialization and finite nonnegative dt");
    SinglePlateCovariance F = SinglePlateCovariance::Identity(), Q = SinglePlateCovariance::Zero();
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    for (int i = 0; i < 6; i += 2)
    {
        F(i, i + 1) = dt;
        Q(i, i) = single_plate_process_noise * dt4 / 4;
        Q(i, i + 1) = Q(i + 1, i) = single_plate_process_noise * dt3 / 2;
        Q(i + 1, i + 1) = single_plate_process_noise * dt2;
    }
    const SinglePlateState predicted = F * self.state_;
    const SinglePlateCovariance covariance = F * self.covariance_ * F.transpose() + Q;
    if (!predicted.allFinite() || !covariance.allFinite())
        throw std::invalid_argument("Non-finite single-plate prediction");
    self.state_ = predicted;
    self.covariance_ = covariance;
}

bool single_plate_update(SinglePlateEKF &self, const SinglePlateObservation &z) {
    if (!self.initialized_ || !single_plate_valid_observation(z) || !single_plate_valid_geometry(self.state_))
        return false;
    const SinglePlateJacobian H = single_plate_jacobian(self.state_);
    SinglePlateObservation residual = z - single_plate_h(self.state_);
    residual(0) = single_plate_wrap_to_pi(residual(0));
    residual(1) = single_plate_wrap_to_pi(residual(1));
    const Eigen::Matrix3d R = SinglePlateObservation(.0016, .0016, .16).asDiagonal(),
                          S = H * self.covariance_ * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix3d> solver(S);
    if (solver.info() != Eigen::Success || !solver.isPositive())
        return false;
    const Eigen::Matrix<double, 6, 3> K = solver.solve(H * self.covariance_).transpose();
    const SinglePlateState corrected = self.state_ + K * residual;
    const SinglePlateCovariance A = SinglePlateCovariance::Identity() - K * H;
    SinglePlateCovariance covariance = A * self.covariance_ * A.transpose() + K * R * K.transpose();
    covariance = ((covariance + covariance.transpose()) * .5).eval();
    if (!corrected.allFinite() || !covariance.allFinite())
        return false;
    self.state_ = corrected;
    self.covariance_ = covariance;
    return true;
}

static bool single_plate_valid_geometry(const SinglePlateState &s) {
    return s.allFinite() && std::hypot(s(0), s(4)) > single_plate_geometry_epsilon &&
           std::hypot(std::hypot(s(0), s(4)), s(2)) > single_plate_geometry_epsilon;
}

// Online model selection and geometry, without file or window operations.
constexpr double missing = std::numeric_limits<double>::quiet_NaN();

static PredictionGeometry polar_geometry(const PolarState &s)
{
    PredictionGeometry geometry;
    geometry.center = Eigen::Vector3d(s(0), s(2), s(4));
    for (int i = 0; i < 4; ++i)
    {
        const double yaw = polar_wrap_to_pi(s(6) + i * polar_pi / 2);
        geometry.plates.emplace_back(s(0) + s(8) * std::sin(yaw), s(2),
                                     s(4) - s(8) * std::cos(yaw), yaw);
    }
    return geometry;
}

static PredictionGeometry armor_geometry(const Forecast &forecast)
{
    PredictionGeometry geometry;
    const auto &s = forecast.state;
    geometry.center = Eigen::Vector3d(s(0), s(2), s(4));
    for (int i = 0; i < 4; ++i)
        geometry.plates.push_back(forecast.plates.row(i).transpose());
    return geometry;
}

VideoPrediction video_predictor_update(VideoPredictor &self, std::int64_t frame, double timestamp, const std::vector<VideoObservation> &obs) {
    if (!std::isfinite(self.horizon_ms) || self.horizon_ms < 0)
        throw std::invalid_argument("Prediction horizon must be finite and nonnegative (ms)");
    if (self.type != PREDICTOR_SINGLE_PLATE && self.type != PREDICTOR_POLAR && self.type != PREDICTOR_ARMOR)
        throw std::invalid_argument("Unknown predictor type");
    if (frame < 0 || frame <= self.last_frame || !std::isfinite(timestamp) || timestamp < 0 ||
        (self.last_frame >= 0 && timestamp <= self.last_timestamp))
        throw std::invalid_argument("Frame IDs and source timestamps must be nonnegative and strictly increasing");
    VideoPrediction result;
    result.horizon_ms = self.horizon_ms;
    const double dt = (timestamp - self.last_timestamp) / 1000;
    self.last_frame = frame;
    self.last_timestamp = timestamp;
    const VideoObservation *selected = nullptr;
    for (const auto &o : obs)
    {
        const bool valid = self.type == PREDICTOR_ARMOR ?
            armor_valid_observation(self.armor, o.measurement) :
            o.position.allFinite() && single_plate_valid_observation(o.measurement.head<3>()) &&
            (self.type == PREDICTOR_SINGLE_PLATE || o.measurement.allFinite());
        if (valid && (!selected || o.measurement(2) < selected->measurement(2)))
            selected = &o;
    }
    std::vector<Observation> all;
    if (self.type == PREDICTOR_ARMOR)
        for (const auto &o : obs)
            all.push_back({o.measurement});
    const bool initialized = self.type == PREDICTOR_SINGLE_PLATE ? self.basic.initialized_ :
                             self.type == PREDICTOR_POLAR ? self.polar.is_initialized : self.armor.is_initialized;
    if (!initialized)
    {
        if (selected)
        {
            const auto &z = selected->measurement;
            if (self.type == PREDICTOR_SINGLE_PLATE)
                single_plate_initialize(self.basic, z.head<3>());
            else if (self.type == PREDICTOR_POLAR)
            {
                const auto &p = selected->position;
                self.polar.X << p(0) - .26 * std::sin(z(3)), 0, p(1), 0,
                    p(2) + .26 * std::cos(z(3)), 0, z(3), 0, .26;
                self.polar.is_initialized = true;
            }
            else
                armor_initialize(self.armor, z);
        }
        if (self.type == PREDICTOR_ARMOR)
        {
            std::vector<Diagnostic> ds(obs.size());
            for (std::size_t i = 0; i < obs.size(); ++i)
            {
                auto &d = ds[i];
                d.observation_index = i;
                d.accepted = &obs[i] == selected;
                d.armor_id = d.accepted ? 0 : -1;
                d.reason = !armor_valid_observation(self.armor, obs[i].measurement) ? "invalid" : d.accepted ? "initialization"
                                                                                                 : "initialization_unused";
            }
            result.diagnostics = std::move(ds);
        }
        result.status = selected ? "initialized" : "waiting";
    }
    else
    {
        bool updated = false;
        if (self.type == PREDICTOR_SINGLE_PLATE)
        {
            single_plate_predict(self.basic, dt);
            const auto &s = self.basic.state_;
            const auto predicted = single_plate_h(s);
            Eigen::Vector4d errors = Eigen::Vector4d::Constant(missing);
            if (selected)
                errors << s(0) - selected->position(0), s(4) - selected->position(2),
                    single_plate_wrap_to_pi(predicted(0) - selected->measurement(0)), predicted(2) - selected->measurement(2);
            result.values = {double(frame), s(0), selected ? selected->position(0) : missing, errors(0),
                             s(4), selected ? selected->position(2) : missing, errors(1), predicted(0),
                             selected ? selected->measurement(0) : missing, errors(2), predicted(2),
                             selected ? selected->measurement(2) : missing, errors(3)};
            result.value_count = 13;
            result.errors = errors;
            updated = selected && single_plate_update(self.basic, selected->measurement.head<3>());
        }
        else if (self.type == PREDICTOR_POLAR)
        {
            polar_predict(self.polar, dt);
            Eigen::Vector4d errors = Eigen::Vector4d::Constant(missing);
            if (selected)
            {
                const int id = polar_round_even(polar_wrap_to_pi(selected->measurement(3) - self.polar.X(6)) /
                                                    (polar_pi / 2));
                errors = polar_angular_residual(polar_h(self.polar.X, id), selected->measurement);
                polar_update(self.polar, selected->measurement);
            }
            const auto &s = self.polar.X;
            result.values = {double(frame), s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7), s(8),
                             errors(0), errors(1), errors(2), errors(3), selected ? selected->measurement(3) : missing};
            result.value_count = 15;
            result.errors = errors;
            updated = selected != nullptr;
        }
        else
        {
            armor_predict(self.armor, dt);
            const ArmorState prior = self.armor.X;
            const auto [matches, ds] = armor_update_multi(self.armor, all);
            result.diagnostics = ds;
            int id = -1;
            Eigen::Vector4d plate = Eigen::Vector4d::Constant(missing), errors = plate;
            double observed_yaw = missing;
            if (!matches.empty())
            {
                const auto &match = *std::min_element(matches.begin(), matches.end(),
                                                      [](const auto &a, const auto &b)
                                                      { return a.Z_obs(2) < b.Z_obs(2); });
                id = match.armor_id;
                plate = armor_plate_pose(prior, id);
                errors = -match.residual;
                observed_yaw = match.Z_obs(3);
            }
            const auto forecast = armor_forecast(self.armor, self.horizon_ms / 1000);
            const auto &f = forecast.state;
            const auto &s = self.armor.X;
            result.values = {double(frame), timestamp, self.horizon_ms, timestamp + self.horizon_ms,
                             f(0), f(2), f(4), f(6), s(0), s(2), s(4), s(1), s(3), s(5), s(7), plate(0), plate(2),
                             double(id), s(6), plate(3), observed_yaw, errors(0), errors(1), errors(2), errors(3),
                             s(8), s(9), s(10), double(all.size()), double(matches.size()), double(all.size() - matches.size())};
            result.value_count = 31;
            result.errors = errors;
            updated = !matches.empty();
        }
        result.status = updated ? "updated" : "prediction_only";
    }
    result.initialized = self.type == PREDICTOR_SINGLE_PLATE ? self.basic.initialized_ :
                         self.type == PREDICTOR_POLAR ? self.polar.is_initialized : self.armor.is_initialized;
    if (result.initialized)
    {
        const double forecast_dt = self.horizon_ms / 1000;
        if (self.type == PREDICTOR_SINGLE_PLATE)
        {
            const auto &s = self.basic.state_;
            result.current.plates.emplace_back(s(0), s(2), s(4), 0);
            result.future.plates.emplace_back(s(0) + forecast_dt * s(1), s(2) + forecast_dt * s(3), s(4) + forecast_dt * s(5), 0);
        }
        else if (self.type == PREDICTOR_POLAR)
        {
            auto future = self.polar.X;
            for (int i : {0, 2, 4, 6})
                future(i) += forecast_dt * future(i + 1);
            future(6) = polar_wrap_to_pi(future(6));
            result.current = polar_geometry(self.polar.X);
            result.future = polar_geometry(future);
        }
        else
        {
            result.current = armor_geometry(armor_forecast(self.armor, 0));
            result.future = armor_geometry(armor_forecast(self.armor, forecast_dt));
        }
    }
    return result;
}

// Prediction CSV and RMSE output.

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

PredictionOutput open_prediction_output(PredictorType model, const std::filesystem::path &directory,
                                        const std::string &suffix) {
    PredictionOutput self{};
    self.type = model;

    if (suffix.empty() || suffix.find_first_of("/\\") != std::string::npos)
        throw std::invalid_argument("Invalid output suffix");
    const char *prefix;
    const char *header;
    switch (self.type)
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
    auto open = [](CsvOutput &self, const std::filesystem::path &path, const char *header) {
        self.stream_.open(path);
        if (!self.stream_)
            throw std::runtime_error("Cannot write " + path.string());
        self.stream_.exceptions(std::ios::badbit | std::ios::failbit);
        self.stream_ << header << '\n'
                << std::setprecision(17);
    };
    std::filesystem::create_directories(directory);
    open(self.results, directory / (std::string(prefix) + "prediction_result_" + suffix + ".csv"), header);
    self.metrics_path = directory / (std::string(prefix) + "rmse_result_" + suffix + ".txt");
    if (self.type == PREDICTOR_ARMOR)
    {
        open(self.logs, directory / ("armor_observation_diagnostics_" + suffix + ".csv"),
                  "frame_id,observation_index,accepted,armor_id,nis,reason,timestamp,observed_distance,"
                  "observed_armor_yaw,best_candidate_id,distance_residual");
        open(self.futures, directory / ("armor_future_prediction_" + suffix + ".csv"),
                     "frame_id,timestamp,prediction_timestamp,prediction_horizon_ms,armor_id,x,y,z,"
                     "armor_orientation_yaw,source_status");
    }

    return self;
}

void prediction_output_write(PredictionOutput &self, std::int64_t frame, double timestamp, const std::vector<VideoObservation> &obs,
                             const VideoPrediction &prediction) {
    if (self.finished) throw std::logic_error("Cannot write finished prediction output");
    auto numbers = [](CsvOutput &self, std::initializer_list<double> values) {
        for (double value : values)
        {
            self.stream_ << (self.first_ ? "" : ",");
            if (std::isfinite(value))
                self.stream_ << value;
            self.first_ = false;
        }
    };
    if (self.type == PREDICTOR_ARMOR) {
        for (const auto &d : prediction.diagnostics) {
            const auto &z = obs.at(d.observation_index).measurement;
            numbers(self.logs, {double(frame), double(d.observation_index)});
            self.logs.stream_ << "," << (d.accepted ? "True" : "False");
            numbers(self.logs, {double(d.armor_id), d.nis});
            self.logs.stream_ << "," << d.reason;
            numbers(self.logs, {timestamp, z(2), z(3), double(d.best_candidate_id), d.distance_residual});
            self.logs.stream_ << '\n';
            self.logs.first_ = true;
        }
    }
    if (!prediction.value_count) return;
    const std::size_t columns = self.type == PREDICTOR_SINGLE_PLATE ? 13 : self.type == PREDICTOR_POLAR ? 15 : 31;
    if (prediction.value_count != columns) throw std::logic_error("Prediction/output model mismatch");
    for (std::size_t i = 0; i < columns; ++i) numbers(self.results, {prediction.values[i]});
    if (self.type == PREDICTOR_ARMOR) {
        self.results.stream_ << "," << prediction.status;
        for (std::size_t i = 0; i < prediction.future.plates.size(); ++i) {
            const auto &p = prediction.future.plates[i];
            numbers(self.futures, {double(frame), timestamp, timestamp + prediction.horizon_ms, prediction.horizon_ms,
                             double(i), p(0), p(1), p(2), p(3)});
            self.futures.stream_ << "," << prediction.status;
            self.futures.stream_ << '\n';
            self.futures.first_ = true;
        }
    }
    self.results.stream_ << '\n';
    self.results.first_ = true;
    for (int i = 0; i < 4; ++i) {
        const double e = prediction.errors(i);
        if (std::isfinite(e)) { self.squared_errors[i] += e * e; ++self.error_counts[i]; }
    }
}

void prediction_output_finish(PredictionOutput &self) {
    if (self.finished)
        return;
    for (auto *csv : {&self.results, &self.logs, &self.futures})
        if (csv->stream_.is_open()) {
            csv->stream_.flush();
            csv->stream_.close();
        }
    std::ofstream metrics(self.metrics_path);
    if (!metrics)
        throw std::runtime_error("Cannot write " + self.metrics_path.string());
    metrics.exceptions(std::ios::badbit | std::ios::failbit);
    if (self.type == PREDICTOR_ARMOR)
        metrics << "Metrics: prior residuals of accepted observations only; not ground-truth error.\n";
    metrics << std::fixed << std::setprecision(6);
    const std::array<const char *, 4> basic_names = {"RMSE_x", "RMSE_z", "RMSE_yaw", "RMSE_distance"},
                                      angular_names = {"RMSE_target_yaw", "RMSE_target_pitch", "RMSE_distance", "RMSE_armor_yaw"},
                                      basic_units = {"m", "m", "rad", "m"}, angular_units = {"rad", "rad", "m", "rad"};
    const auto &names = self.type == PREDICTOR_SINGLE_PLATE ? basic_names : angular_names;
    const auto &units = self.type == PREDICTOR_SINGLE_PLATE ? basic_units : angular_units;
    for (std::size_t i = 0; i < names.size(); ++i)
        metrics << names[i] << ": " << (self.error_counts[i] ? std::sqrt(self.squared_errors[i] / double(self.error_counts[i])) : missing)
                << ' ' << units[i] << '\n';
    metrics.close();
    self.finished = true;
}
