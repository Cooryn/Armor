#include "predictor.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <initializer_list>
#include <stdexcept>

static bool single_plate_valid_geometry(const Eigen::Matrix<double, 6, 1> &s);

double single_plate_wrap_to_pi(double angle)
{
    double result = std::fmod(angle + single_plate_pi, 2 * single_plate_pi);
    return (result < 0 ? result + 2 * single_plate_pi : result) - single_plate_pi;
}

bool single_plate_valid_observation(const Eigen::Vector3d &z)
{
    return z.allFinite() && z(2) > single_plate_geometry_epsilon && std::abs(z(1)) < single_plate_pi / 2 &&
           z(2) * std::cos(z(1)) > single_plate_geometry_epsilon;
}

Eigen::Vector3d single_plate_h(const Eigen::Matrix<double, 6, 1> &s)
{
    const double horizontal = std::hypot(s(0), s(4));
    return {std::atan2(s(0), s(4)), std::atan2(s(2), horizontal), std::hypot(horizontal, s(2))};
}

Eigen::Matrix<double, 3, 6> single_plate_jacobian(const Eigen::Matrix<double, 6, 1> &s)
{
    if (!single_plate_valid_geometry(s))
        throw std::invalid_argument("Singular single-plate observation geometry");
    const double x = s(0), y = s(2), z = s(4), horizontal = std::hypot(x, z),
                 distance = std::hypot(horizontal, y), horizontal2 = horizontal * horizontal,
                 distance2 = distance * distance;
    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
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

void SinglePlateEKF::initialize(const Eigen::Vector3d &z)
{
    if (!single_plate_valid_observation(z))
        throw std::invalid_argument("Invalid single-plate initialization observation");
    const double horizontal = z(2) * std::cos(z(1));
    state_ << horizontal * std::sin(z(0)), 0, z(2) * std::sin(z(1)), 0,
        horizontal * std::cos(z(0)), 0;
    covariance_ = Eigen::Matrix<double, 6, 6>::Identity() * 10;
    initialized_ = true;
}

void SinglePlateEKF::predict(double dt)
{
    if (!initialized_ || !std::isfinite(dt) || dt < 0)
        throw std::invalid_argument("Prediction requires initialization and finite nonnegative dt");
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity(), Q = Eigen::Matrix<double, 6, 6>::Zero();
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    for (int i = 0; i < 6; i += 2)
    {
        F(i, i + 1) = dt;
        Q(i, i) = single_plate_process_noise * dt4 / 4;
        Q(i, i + 1) = Q(i + 1, i) = single_plate_process_noise * dt3 / 2;
        Q(i + 1, i + 1) = single_plate_process_noise * dt2;
    }
    const Eigen::Matrix<double, 6, 1> predicted = F * state_;
    const Eigen::Matrix<double, 6, 6> covariance = F * covariance_ * F.transpose() + Q;
    if (!predicted.allFinite() || !covariance.allFinite())
        throw std::invalid_argument("Non-finite single-plate prediction");
    state_ = predicted;
    covariance_ = covariance;
}

bool SinglePlateEKF::update(const Eigen::Vector3d &z)
{
    if (!initialized_ || !single_plate_valid_observation(z) || !single_plate_valid_geometry(state_))
        return false;
    const Eigen::Matrix<double, 3, 6> H = single_plate_jacobian(state_);
    Eigen::Vector3d residual = z - single_plate_h(state_);
    residual(0) = single_plate_wrap_to_pi(residual(0));
    residual(1) = single_plate_wrap_to_pi(residual(1));
    const Eigen::Matrix3d R = Eigen::Vector3d(.0016, .0016, .16).asDiagonal(),
                          S = H * covariance_ * H.transpose() + R;
    const Eigen::LDLT<Eigen::Matrix3d> solver(S);
    if (solver.info() != Eigen::Success || !solver.isPositive())
        return false;
    const Eigen::Matrix<double, 6, 3> K = solver.solve(H * covariance_).transpose();
    const Eigen::Matrix<double, 6, 1> corrected = state_ + K * residual;
    const Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Identity() - K * H;
    Eigen::Matrix<double, 6, 6> covariance = A * covariance_ * A.transpose() + K * R * K.transpose();
    covariance = ((covariance + covariance.transpose()) * .5).eval();
    if (!corrected.allFinite() || !covariance.allFinite())
        return false;
    state_ = corrected;
    covariance_ = covariance;
    return true;
}

static bool single_plate_valid_geometry(const Eigen::Matrix<double, 6, 1> &s)
{
    return s.allFinite() && std::hypot(s(0), s(4)) > single_plate_geometry_epsilon &&
           std::hypot(std::hypot(s(0), s(4)), s(2)) > single_plate_geometry_epsilon;
}

// Online model selection and geometry, without file or window operations.
constexpr double missing = std::numeric_limits<double>::quiet_NaN();

static PredictionGeometry polar_geometry(const Eigen::Matrix<double, 9, 1> &s)
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

VideoPrediction VideoPredictor::update(std::int64_t frame, double timestamp, const std::vector<VideoObservation> &obs)
{
    if (!std::isfinite(horizon_ms) || horizon_ms < 0)
        throw std::invalid_argument("Prediction horizon must be finite and nonnegative (ms)");
    if (type != PREDICTOR_SINGLE_PLATE && type != PREDICTOR_POLAR && type != PREDICTOR_ARMOR)
        throw std::invalid_argument("Unknown predictor type");
    if (frame < 0 || frame <= last_frame || !std::isfinite(timestamp) || timestamp < 0 ||
        (last_frame >= 0 && timestamp <= last_timestamp))
        throw std::invalid_argument("Frame IDs and source timestamps must be nonnegative and strictly increasing");
    VideoPrediction result;
    result.horizon_ms = horizon_ms;
    const double dt = (timestamp - last_timestamp) / 1000;
    last_frame = frame;
    last_timestamp = timestamp;
    const VideoObservation *selected = nullptr;
    for (const auto &o : obs)
    {
        const bool valid = type == PREDICTOR_ARMOR ? armor.valid_observation(o.measurement) : o.position.allFinite() && single_plate_valid_observation(o.measurement.head<3>()) && (type == PREDICTOR_SINGLE_PLATE || o.measurement.allFinite());
        if (valid && (!selected || o.measurement(2) < selected->measurement(2)))
            selected = &o;
    }
    std::vector<Observation> all;
    if (type == PREDICTOR_ARMOR)
        for (const auto &o : obs)
            all.push_back({o.measurement});
    const bool initialized = type == PREDICTOR_SINGLE_PLATE ? basic.initialized_ : type == PREDICTOR_POLAR ? polar.is_initialized
                                                                                                           : armor.is_initialized;
    if (!initialized)
    {
        if (selected)
        {
            const auto &z = selected->measurement;
            if (type == PREDICTOR_SINGLE_PLATE)
                basic.initialize(z.head<3>());
            else if (type == PREDICTOR_POLAR)
            {
                const auto &p = selected->position;
                polar.X << p(0) - .26 * std::sin(z(3)), 0, p(1), 0,
                    p(2) + .26 * std::cos(z(3)), 0, z(3), 0, .26;
                polar.is_initialized = true;
            }
            else
                armor.initialize(z);
        }
        if (type == PREDICTOR_ARMOR)
        {
            std::vector<Diagnostic> ds(obs.size());
            for (std::size_t i = 0; i < obs.size(); ++i)
            {
                auto &d = ds[i];
                d.observation_index = i;
                d.accepted = &obs[i] == selected;
                d.armor_id = d.accepted ? 0 : -1;
                d.reason = !armor.valid_observation(obs[i].measurement) ? "invalid" : d.accepted ? "initialization"
                                                                                                 : "initialization_unused";
            }
            result.diagnostics = std::move(ds);
        }
        result.status = selected ? "initialized" : "waiting";
    }
    else
    {
        bool updated = false;
        if (type == PREDICTOR_SINGLE_PLATE)
        {
            basic.predict(dt);
            const auto &s = basic.state_;
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
            updated = selected && basic.update(selected->measurement.head<3>());
        }
        else if (type == PREDICTOR_POLAR)
        {
            polar.predict(dt);
            Eigen::Vector4d errors = Eigen::Vector4d::Constant(missing);
            if (selected)
            {
                const int id = polar_round_even(polar_wrap_to_pi(selected->measurement(3) - polar.X(6)) /
                                                (polar_pi / 2));
                errors = polar_angular_residual(polar_h(polar.X, id), selected->measurement);
                polar.update(selected->measurement);
            }
            const auto &s = polar.X;
            result.values = {double(frame), s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7), s(8),
                             errors(0), errors(1), errors(2), errors(3), selected ? selected->measurement(3) : missing};
            result.value_count = 15;
            result.errors = errors;
            updated = selected != nullptr;
        }
        else
        {
            armor.predict(dt);
            const Eigen::Matrix<double, 11, 1> prior = armor.X;
            const auto [matches, ds] = armor.update_multi(all);
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
            const auto forecast = armor.forecast(horizon_ms / 1000);
            const auto &f = forecast.state;
            const auto &s = armor.X;
            result.values = {double(frame), timestamp, horizon_ms, timestamp + horizon_ms,
                             f(0), f(2), f(4), f(6), s(0), s(2), s(4), s(1), s(3), s(5), s(7), plate(0), plate(2),
                             double(id), s(6), plate(3), observed_yaw, errors(0), errors(1), errors(2), errors(3),
                             s(8), s(9), s(10), double(all.size()), double(matches.size()), double(all.size() - matches.size())};
            result.value_count = 31;
            result.errors = errors;
            updated = !matches.empty();
        }
        result.status = updated ? "updated" : "prediction_only";
    }
    result.initialized = type == PREDICTOR_SINGLE_PLATE ? basic.initialized_ : type == PREDICTOR_POLAR ? polar.is_initialized
                                                                                                       : armor.is_initialized;
    if (result.initialized)
    {
        const double forecast_dt = horizon_ms / 1000;
        if (type == PREDICTOR_SINGLE_PLATE)
        {
            const auto &s = basic.state_;
            result.current.plates.emplace_back(s(0), s(2), s(4), 0);
            result.future.plates.emplace_back(s(0) + forecast_dt * s(1), s(2) + forecast_dt * s(3), s(4) + forecast_dt * s(5), 0);
        }
        else if (type == PREDICTOR_POLAR)
        {
            auto future = polar.X;
            for (int i : {0, 2, 4, 6})
                future(i) += forecast_dt * future(i + 1);
            future(6) = polar_wrap_to_pi(future(6));
            result.current = polar_geometry(polar.X);
            result.future = polar_geometry(future);
        }
        else
        {
            result.current = armor_geometry(armor.forecast(0));
            result.future = armor_geometry(armor.forecast(forecast_dt));
        }
    }
    return result;
}

// Prediction CSV and RMSE output.

void PredictionOutput::write(std::int64_t frame, double timestamp, const std::vector<VideoObservation> &obs,
                             const VideoPrediction &prediction)
{
    if (finished)
        throw std::logic_error("Cannot write finished prediction output");
    auto numbers = [](CsvOutput &csv, std::initializer_list<double> values)
    {
        for (double value : values)
        {
            csv.stream_ << (csv.first_ ? "" : ",");
            if (std::isfinite(value))
                csv.stream_ << value;
            csv.first_ = false;
        }
    };
    if (type == PREDICTOR_ARMOR)
    {
        for (const auto &d : prediction.diagnostics)
        {
            const auto &z = obs.at(d.observation_index).measurement;
            numbers(logs, {double(frame), double(d.observation_index)});
            logs.stream_ << "," << (d.accepted ? "True" : "False");
            numbers(logs, {double(d.armor_id), d.nis});
            logs.stream_ << "," << d.reason;
            numbers(logs, {timestamp, z(2), z(3), double(d.best_candidate_id), d.distance_residual});
            logs.stream_ << '\n';
            logs.first_ = true;
        }
    }
    if (!prediction.value_count)
        return;
    const std::size_t columns = type == PREDICTOR_SINGLE_PLATE ? 13 : type == PREDICTOR_POLAR ? 15
                                                                                              : 31;
    if (prediction.value_count != columns)
        throw std::logic_error("Prediction/output model mismatch");
    for (std::size_t i = 0; i < columns; ++i)
        numbers(results, {prediction.values[i]});
    if (type == PREDICTOR_ARMOR)
    {
        results.stream_ << "," << prediction.status;
        for (std::size_t i = 0; i < prediction.future.plates.size(); ++i)
        {
            const auto &p = prediction.future.plates[i];
            numbers(futures, {double(frame), timestamp, timestamp + prediction.horizon_ms, prediction.horizon_ms,
                              double(i), p(0), p(1), p(2), p(3)});
            futures.stream_ << "," << prediction.status;
            futures.stream_ << '\n';
            futures.first_ = true;
        }
    }
    results.stream_ << '\n';
    results.first_ = true;
    for (int i = 0; i < 4; ++i)
    {
        const double e = prediction.errors(i);
        if (std::isfinite(e))
        {
            squared_errors[i] += e * e;
            ++error_counts[i];
        }
    }
}

void PredictionOutput::finish()
{
    if (finished)
        return;
    for (auto *csv : {&results, &logs, &futures})
        if (csv->stream_.is_open())
        {
            csv->stream_.flush();
            csv->stream_.close();
        }
    std::ofstream metrics(metrics_path);
    if (!metrics)
        throw std::runtime_error("Cannot write " + metrics_path.string());
    metrics.exceptions(std::ios::badbit | std::ios::failbit);
    if (type == PREDICTOR_ARMOR)
        metrics << "Metrics: prior residuals of accepted observations only; not ground-truth error.\n";
    metrics << std::fixed << std::setprecision(6);
    const std::array<const char *, 4> basic_names = {"RMSE_x", "RMSE_z", "RMSE_yaw", "RMSE_distance"},
                                      angular_names = {"RMSE_target_yaw", "RMSE_target_pitch", "RMSE_distance", "RMSE_armor_yaw"},
                                      basic_units = {"m", "m", "rad", "m"}, angular_units = {"rad", "rad", "m", "rad"};
    const auto &names = type == PREDICTOR_SINGLE_PLATE ? basic_names : angular_names;
    const auto &units = type == PREDICTOR_SINGLE_PLATE ? basic_units : angular_units;
    for (std::size_t i = 0; i < names.size(); ++i)
        metrics << names[i] << ": " << (error_counts[i] ? std::sqrt(squared_errors[i] / double(error_counts[i])) : missing)
                << ' ' << units[i] << '\n';
    metrics.close();
    finished = true;
}
