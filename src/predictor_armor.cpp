#include "predictor_armor.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>

namespace {
namespace fs = std::filesystem;
using predictor::ArmorEKF;
using predictor::Diagnostic;
using predictor::Observation;
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
constexpr std::array<const char *, 6> input_columns = {
    "frame_id", "timestamp", "target_yaw", "target_pitch", "distance", "armor_orientation_yaw"};
constexpr std::array<const char *, 4> origin_columns = {
    "base_reference_timestamp_ms", "base_origin_x_m", "base_origin_y_m", "base_origin_z_m"};
struct Frame {
    double timestamp;
    std::vector<Observation> observations;
};
struct Input {
    std::map<long long, Frame> frames;
    bool base = false;
    std::array<double, 4> origin{};
};

bool read_record(std::istream &input, std::vector<std::string> &fields) {
    fields.clear();
    std::string field;
    bool quoted = false;
    char c;
    while (input.get(c)) {
        if (c == '"') {
            if (quoted && input.peek() == '"') {
                input.get(c);
                field += '"';
            } else
                quoted = !quoted;
        } else if (!quoted && (c == ',' || c == '\n' || c == '\r')) {
            fields.push_back(std::move(field));
            field.clear();
            if (c != ',') {
                if (c == '\r' && input.peek() == '\n') input.get(c);
                if (fields.size() > 1 || !fields[0].empty()) return true;
                fields.clear();
            }
        } else
            field += c;
    }
    if (quoted) throw std::invalid_argument("Unclosed CSV quote");
    if (input.bad()) throw std::runtime_error("CSV read failed");
    if (fields.empty() && field.empty()) return false;
    fields.push_back(std::move(field));
    return true;
}
double number(const std::string &text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return missing;
    const auto value = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
    if (value == "NA" || value == "N/A" || value == "NULL" || value == "null" ||
        value == "None" || value == "NaN" || value == "nan" || value == "<NA>") return missing;
    size_t parsed;
    const double result = std::stod(value, &parsed);
    if (parsed != value.size()) throw std::invalid_argument("Invalid numeric CSV value: " + value);
    return result;
}
Input load(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + path.string());
    std::vector<std::string> header, row;
    if (!read_record(stream, header)) throw std::invalid_argument("No columns to parse from file");
    if (header[0].compare(0, 3, "\xef\xbb\xbf") == 0) header[0].erase(0, 3);
    auto column = [&](const char *name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) throw std::invalid_argument(std::string("Missing CSV column: ") + name);
        return static_cast<size_t>(it - header.begin());
    };
    std::array<size_t, 6> indices{};
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = column(input_columns[i]);
    const auto coordinate = std::find(header.begin(), header.end(), "coordinate_frame");
    const size_t frame_column = static_cast<size_t>(coordinate - header.begin());
    std::array<size_t, 4> origin_indices{};
    Input data;
    bool first_row = true;
    while (read_record(stream, row)) {
        auto cell = [&](size_t index) { return index < row.size() ? row[index] : std::string(); };
        std::array<double, 6> values{};
        for (size_t i = 0; i < values.size(); ++i) values[i] = number(cell(indices[i]));
        if (!std::isfinite(values[0]) || values[0] < 0 || std::floor(values[0]) != values[0] ||
            values[0] >= double(std::numeric_limits<long long>::max()) ||
            !std::isfinite(values[1]) || values[1] < 0)
            throw std::invalid_argument("Invalid frame_id or timestamp");
        // The detector's legacy CSV has no tag; its documented frame is camera.
        const std::string frame = coordinate == header.end() ? "camera" : cell(frame_column);
        if (frame != "camera" && frame != "base") throw std::invalid_argument("Unknown coordinate_frame");
        if (!first_row && data.base != (frame == "base")) throw std::invalid_argument("Mixed coordinate frames");
        data.base = frame == "base";
        if (data.base) {
            if (first_row)
                for (size_t i = 0; i < origin_indices.size(); ++i) origin_indices[i] = column(origin_columns[i]);
            for (size_t i = 0; i < data.origin.size(); ++i) {
                const double value = number(cell(origin_indices[i]));
                if (!std::isfinite(value) || (!first_row && value != data.origin[i]) || (i == 0 && value < 0))
                    throw std::invalid_argument("Missing or mixed base origin metadata; regenerate base CSV");
                data.origin[i] = value;
            }
        }
        first_row = false;
        auto [it, inserted] = data.frames.try_emplace(static_cast<long long>(values[0]), Frame{values[1], {}});
        if (!inserted && it->second.timestamp != values[1])
            throw std::invalid_argument("All observations of a frame must share its timestamp");
        it->second.observations.push_back({Eigen::Vector4d(values[2], values[3], values[4], values[5])});
    }
    for (auto it = data.frames.begin(); it != data.frames.end(); ++it)
        if (it != data.frames.begin() && it->second.timestamp <= std::prev(it)->second.timestamp)
            throw std::invalid_argument("Timestamps must increase between frames");
    return data;
}

// Fixed schemas: numbers and the program's own status/reason strings only.
class CsvOutput {
    std::ofstream stream_;
    const Input &data_;
    bool first_ = true;
  public:
    CsvOutput(const fs::path &path, const char *header, const Input &data, bool quaternion)
        : stream_(path), data_(data) {
        if (!stream_) throw std::runtime_error("Cannot write " + path.string());
        stream_ << header;
        if (data.base) {
            stream_ << ",coordinate_frame";
            for (const auto *name : origin_columns) stream_ << ',' << name;
            if (quaternion) stream_ << ",qw,qx,qy,qz";
        }
        stream_ << '\n' << std::setprecision(17);
    }
    void text(const std::string &value) {
        stream_ << (first_ ? "" : ",") << value;
        first_ = false;
    }
    void numbers(std::initializer_list<double> values) {
        for (double value : values) {
            stream_ << (first_ ? "" : ",");
            if (!std::isnan(value)) stream_ << value;
            first_ = false;
        }
    }
    void metadata() {
        if (data_.base) {
            text("base");
            numbers({data_.origin[0], data_.origin[1], data_.origin[2], data_.origin[3]});
        }
    }
    void end_row() { stream_ << '\n'; first_ = true; }
    void finish() {
        stream_.flush();
        if (!stream_) throw std::runtime_error("CSV write failed");
    }
};
void write_diagnostics(CsvOutput &out, double frame, double timestamp,
                       const std::vector<Observation> &obs, const std::vector<Diagnostic> &diagnostics) {
    for (const auto &d : diagnostics) {
        const auto &z = obs[d.observation_index].Z_obs;
        out.numbers({frame, double(d.observation_index)});
        out.text(d.accepted ? "True" : "False");
        out.numbers({double(d.armor_id), d.nis});
        out.text(d.reason);
        out.numbers({timestamp, z(2), z(3), double(d.best_candidate_id), d.distance_residual});
        out.metadata();
        out.end_row();
    }
}
void run_predict_armor(const fs::path &path, const fs::path &output, const std::string &suffix, double horizon) {
    const Input data = load(path);
    ArmorEKF filter;
    filter.base_frame = data.base;
    auto usable = [&](const auto &frame) {
        return std::any_of(frame.second.observations.begin(), frame.second.observations.end(),
                           [&](const auto &o) { return filter.valid_observation(o.Z_obs); });
    };
    const auto first_valid = std::find_if(data.frames.begin(), data.frames.end(), usable);
    if (data.frames.size() < 2 || first_valid == data.frames.end() || first_valid == std::prev(data.frames.end())) {
        std::cout << "No prediction exported: at least two detected frames are required.\n";
        return;
    }
    fs::create_directories(output);
    CsvOutput results(output / ("armor_prediction_result_" + suffix + ".csv"),
        "frame_id,timestamp,prediction_horizon_ms,prediction_timestamp,future_xc,future_yc,future_zc,"
        "future_body_yaw,xc,yc,zc,vxc,vyc,vzc,w,xa,za,armor_id,body_yaw,pred_armor_yaw,obs_armor_yaw,"
        "err_target_yaw,err_target_pitch,err_distance,err_armor_yaw,r,dl,dh,observation_count,accepted_count,"
        "rejected_count,status", data, false);
    CsvOutput logs(output / ("armor_observation_diagnostics_" + suffix + ".csv"),
        "frame_id,observation_index,accepted,armor_id,nis,reason,timestamp,observed_distance,"
        "observed_armor_yaw,best_candidate_id,distance_residual", data, false);
    CsvOutput futures(output / ("armor_future_prediction_" + suffix + ".csv"),
        "frame_id,timestamp,prediction_timestamp,prediction_horizon_ms,armor_id,x,y,z,"
        "armor_orientation_yaw,source_status", data, true);
    std::array<double, 4> squared_errors{};
    std::array<size_t, 4> error_counts{};
    double last = 0;
    const std::vector<Observation> empty;
    auto right = data.frames.begin();
    for (long long frame = right->first, end = data.frames.rbegin()->first; frame <= end; ++frame) {
        while (right->first < frame) ++right;
        double timestamp = right->second.timestamp;
        if (right->first != frame) {
            const auto left = std::prev(right);
            timestamp = left->second.timestamp + (timestamp - left->second.timestamp) *
                double(frame - left->first) / double(right->first - left->first);
        }
        const auto &obs = right->first == frame ? right->second.observations : empty;
        const double id = static_cast<double>(frame);
        if (!filter.is_initialized) {
            int seed = -1;
            for (size_t i = 0; i < obs.size(); ++i)
                if (filter.valid_observation(obs[i].Z_obs) && (seed < 0 || obs[i].Z_obs(2) < obs[seed].Z_obs(2)))
                    seed = static_cast<int>(i);
            if (seed >= 0) {
                filter.initialize(obs[seed].Z_obs);
                last = timestamp;
            }
            std::vector<Diagnostic> diagnostics(obs.size());
            for (size_t i = 0; i < obs.size(); ++i) {
                auto &d = diagnostics[i];
                d.observation_index = i;
                d.accepted = static_cast<int>(i) == seed;
                d.armor_id = d.accepted ? 0 : -1;
                d.reason = seed < 0 ? "invalid" : (d.accepted ? "initialization" : "initialization_unused");
            }
            write_diagnostics(logs, id, timestamp, obs, diagnostics);
            continue;
        }
        filter.predict((timestamp - last) / 1000);
        last = timestamp;
        const ArmorEKF::State prior = filter.X;
        const auto [matches, diagnostics] = filter.update_multi(obs);
        write_diagnostics(logs, id, timestamp, obs, diagnostics);
        int armor_id = -1;
        Eigen::Vector4d plate = Eigen::Vector4d::Constant(missing), errors = plate;
        double observed_yaw = missing;
        if (!matches.empty()) {
            const auto &match = *std::min_element(matches.begin(), matches.end(),
                [](const auto &a, const auto &b) { return a.Z_obs(2) < b.Z_obs(2); });
            armor_id = match.armor_id;
            plate = filter.plate_pose(prior, armor_id);
            errors = -match.residual;
            observed_yaw = match.Z_obs(3);
        }
        const auto forecast = filter.forecast(horizon / 1000);
        const auto &f = forecast.state;
        const auto &s = filter.X;
        const std::string status = matches.empty() ? "prediction_only" : "updated";
        for (int i = 0; i < 4; ++i) {
            const double yaw = forecast.plates(i, 3);
            futures.numbers({id, timestamp, timestamp + horizon, horizon, double(i), forecast.plates(i, 0),
                             forecast.plates(i, 1), forecast.plates(i, 2), yaw});
            futures.text(status);
            futures.metadata();
            if (data.base) futures.numbers({std::cos(yaw / 2), 0, -std::sin(yaw / 2), 0});
            futures.end_row();
        }
        results.numbers({id, timestamp, horizon, timestamp + horizon, f(0), f(2), f(4), f(6),
                        s(0), s(2), s(4), s(1), s(3), s(5), s(7), plate(0), plate(2), double(armor_id),
                        s(6), plate(3), observed_yaw, errors(0), errors(1), errors(2), errors(3),
                        s(8), s(9), s(10), double(obs.size()), double(matches.size()), double(obs.size() - matches.size())});
        results.text(status);
        results.metadata();
        results.end_row();
        for (size_t i = 0; i < squared_errors.size(); ++i)
            if (!std::isnan(errors(static_cast<Eigen::Index>(i)))) {
                squared_errors[i] += errors(static_cast<Eigen::Index>(i)) * errors(static_cast<Eigen::Index>(i));
                ++error_counts[i];
            }
    }
    results.finish();
    logs.finish();
    futures.finish();
    std::ofstream metrics(output / ("armor_rmse_result_" + suffix + ".txt"));
    if (!metrics) throw std::runtime_error("Cannot write RMSE file");
    metrics << "Metrics: prior residuals of accepted observations only; not ground-truth error.\n"
            << std::fixed << std::setprecision(6);
    constexpr std::array<const char *, 4> names = {"RMSE_target_yaw", "RMSE_target_pitch", "RMSE_distance", "RMSE_armor_yaw"},
                                         units = {"rad", "rad", "m", "rad"};
    for (size_t i = 0; i < names.size(); ++i)
        metrics << names[i] << ": " << (error_counts[i] ? std::sqrt(squared_errors[i] / double(error_counts[i])) : missing)
                << ' ' << units[i] << '\n';
    metrics.flush();
    if (!metrics) throw std::runtime_error("RMSE write failed");
}
} // namespace

#ifndef PREDICTOR_DEFAULT_ROOT
#define PREDICTOR_DEFAULT_ROOT "."
#endif
int main(int argc, char **argv) {
    std::string suffix = "1";
    fs::path root = PREDICTOR_DEFAULT_ROOT, input, output;
    double horizon = 50;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            auto value = [&]() {
                if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + argument);
                return std::string(argv[++i]);
            };
            if (argument == "--suffix") suffix = value();
            else if (argument == "--input") input = fs::u8path(value());
            else if (argument == "--output-dir") output = fs::u8path(value());
            else if (argument == "--root") root = fs::u8path(value());
            else if (argument == "--prediction-horizon-ms") horizon = number(value());
            else if (argument == "--help" || argument == "-h") {
                std::cout << "Options: --suffix VALUE --input CSV --output-dir DIR --prediction-horizon-ms MS --root DIR\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (!std::isfinite(horizon) || horizon < 0)
            throw std::invalid_argument("--prediction-horizon-ms must be finite and nonnegative");
        if (input.empty()) input = root / "data" / ("pose_raw_" + suffix + ".csv");
        if (output.empty()) output = root / "results";
        if (!fs::is_regular_file(input)) throw std::invalid_argument("Input CSV does not exist or is not a file: " + input.string());
        run_predict_armor(input, output, suffix, horizon);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
