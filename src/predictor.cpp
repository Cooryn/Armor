#include "predictor.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using predictor::SinglePlateEKF;
constexpr std::array<const char *, 8> input_columns = {
    "frame_id", "timestamp", "x", "y", "z", "target_yaw", "target_pitch", "distance"};
constexpr std::array<const char *, 4> origin_columns = {
    "base_reference_timestamp_ms", "base_origin_x_m", "base_origin_y_m", "base_origin_z_m"};
struct Sample {
    double timestamp;
    Eigen::Vector3d position;
    SinglePlateEKF::Observation observation;
};
struct Input {
    std::map<double, Sample> frames;
    bool base = false;
    std::array<double, 4> origin{};
};

// Read one record, including quoted commas/newlines and doubled quotes.
bool read_record(std::istream &input, std::vector<std::string> &fields) {
    fields.clear();
    std::string field;
    bool in_quote = false;
    char c;
    while (input.get(c)) {
        if (c == '"') {
            if (in_quote && input.peek() == '"') {
                input.get(c);
                field += '"';
            } else
                in_quote = !in_quote;
        } else if (!in_quote && (c == ',' || c == '\n' || c == '\r')) {
            fields.push_back(std::move(field));
            field.clear();
            if (c != ',') {
                if (c == '\r' && input.peek() == '\n')
                    input.get(c);
                if (fields.size() > 1 || !fields[0].empty())
                    return true;
                fields.clear();
            }
        } else
            field += c;
    }
    if (in_quote)
        throw std::invalid_argument("Unclosed CSV quote");
    if (input.bad())
        throw std::runtime_error("CSV read failed");
    if (fields.empty() && field.empty())
        return false;
    fields.push_back(std::move(field));
    return true;
}
double number(const std::string &text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::numeric_limits<double>::quiet_NaN();
    const auto value = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
    if (value == "NA" || value == "N/A" || value == "NULL" || value == "null" ||
        value == "None" || value == "NaN" || value == "nan" || value == "<NA>")
        return std::numeric_limits<double>::quiet_NaN();
    size_t parsed;
    const double result = std::stod(value, &parsed);
    if (parsed != value.size())
        throw std::invalid_argument("Invalid numeric CSV value: " + value);
    return result;
}
Input load(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot read " + path.string());
    std::vector<std::string> header, row;
    if (!read_record(stream, header))
        throw std::invalid_argument("No columns to parse from file");
    if (header[0].compare(0, 3, "\xef\xbb\xbf") == 0)
        header[0].erase(0, 3);
    auto column = [&](const char *name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end())
            throw std::invalid_argument(std::string("Missing CSV column: ") + name);
        return static_cast<size_t>(it - header.begin());
    };
    std::array<size_t, 8> indices{};
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = column(input_columns[i]);
    const auto coordinate = std::find(header.begin(), header.end(), "coordinate_frame");
    const size_t frame_column = static_cast<size_t>(coordinate - header.begin());
    std::array<size_t, 4> origin_indices{};
    Input data;
    bool first_row = true;
    while (read_record(stream, row)) {
        auto cell = [&](size_t index) { return index < row.size() ? row[index] : std::string(); };
        std::array<double, 8> values{};
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = number(cell(indices[i]));
        if (!std::isfinite(values[0]) || values[0] < 0 || std::floor(values[0]) != values[0] ||
            !std::isfinite(values[1]) || values[1] < 0)
            throw std::invalid_argument("Invalid frame_id or timestamp");
        const std::string frame = coordinate == header.end() ? "camera" : cell(frame_column);
        if (frame != "camera" && frame != "base")
            throw std::invalid_argument("Unknown input coordinate_frame");
        if (!first_row && data.base != (frame == "base"))
            throw std::invalid_argument("Mixed coordinate frames in input");
        data.base = frame == "base";
        if (data.base) {
            if (first_row)
                for (size_t i = 0; i < origin_indices.size(); ++i)
                    origin_indices[i] = column(origin_columns[i]);
            for (size_t i = 0; i < data.origin.size(); ++i) {
                const double value = number(cell(origin_indices[i]));
                if (!std::isfinite(value) || (!first_row && value != data.origin[i]) || (i == 0 && value < 0))
                    throw std::invalid_argument("Missing or mixed base origin metadata; regenerate base CSV");
                data.origin[i] = value;
            }
        }
        first_row = false;
        const Sample sample{values[1], {values[2], values[3], values[4]}, {values[5], values[6], values[7]}};
        if (!sample.position.allFinite() || !SinglePlateEKF::valid_observation(sample.observation))
            continue;
        auto [it, inserted] = data.frames.try_emplace(values[0], sample);
        if (!inserted && sample.observation(2) < it->second.observation(2))
            it->second = sample;
    }
    return data;
}
void run_predict(const fs::path &input, const fs::path &output, const std::string &suffix) {
    const Input data = load(input);
    if (data.frames.size() < 2) {
        std::cout << "No prediction exported: at least two detected frames are required.\n";
        return;
    }
    fs::create_directories(output);
    const fs::path csv_path = output / ("prediction_result_" + suffix + ".csv"),
                   metrics_path = output / ("rmse_result_" + suffix + ".txt");
    std::ofstream csv(csv_path);
    if (!csv)
        throw std::runtime_error("Cannot write " + csv_path.string());
    csv << "frame_id,predicted_x,observed_x,error_x,predicted_z,observed_z,error_z,"
           "predicted_yaw,observed_yaw,error_yaw,predicted_distance,observed_distance,error_distance";
    if (data.base) {
        csv << ",coordinate_frame";
        for (const auto *name : origin_columns)
            csv << ',' << name;
    }
    csv << '\n' << std::setprecision(17);
    SinglePlateEKF filter;
    double last_timestamp = 0;
    std::array<double, 4> squared_errors{};
    for (const auto &[frame_id, sample] : data.frames) {
        if (!filter.initialized()) {
            filter.initialize(sample.observation);
            last_timestamp = sample.timestamp;
            continue;
        }
        double dt = (sample.timestamp - last_timestamp) / 1000;
        if (dt <= 0)
            dt = 1. / 30;
        last_timestamp = sample.timestamp;
        filter.predict(dt);
        const auto &s = filter.state();
        const auto predicted = SinglePlateEKF::h(s);
        const std::array<double, 4> errors = {
            s(0) - sample.position(0), s(4) - sample.position(2),
            SinglePlateEKF::wrap_to_pi(predicted(0) - sample.observation(0)),
            predicted(2) - sample.observation(2)};
        const std::array<double, 13> result = {
            frame_id, s(0), sample.position(0), errors[0], s(4), sample.position(2), errors[1],
            predicted(0), sample.observation(0), errors[2], predicted(2), sample.observation(2), errors[3]};
        for (size_t i = 0; i < result.size(); ++i)
            csv << (i ? "," : "") << result[i];
        if (data.base) {
            csv << ",base";
            for (double value : data.origin)
                csv << ',' << value;
        }
        csv << '\n';
        for (size_t i = 0; i < errors.size(); ++i)
            squared_errors[i] += errors[i] * errors[i];
        filter.update(sample.observation);
    }
    std::ofstream metrics(metrics_path);
    if (!metrics)
        throw std::runtime_error("Cannot write " + metrics_path.string());
    constexpr std::array<const char *, 4> names = {"RMSE_x", "RMSE_z", "RMSE_yaw", "RMSE_distance"},
                                         units = {"m", "m", "rad", "m"};
    metrics << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < names.size(); ++i)
        metrics << names[i] << ": " << std::sqrt(squared_errors[i] / static_cast<double>(data.frames.size() - 1))
                << ' ' << units[i] << '\n';
    csv.flush();
    metrics.flush();
    if (!csv || !metrics)
        throw std::runtime_error("Prediction output write failed");
    std::cout << "已输出预测结果: " << csv_path.string() << '\n'
              << "已计算RMSE: " << metrics_path.string() << '\n';
}
} // namespace

#ifndef PREDICTOR_DEFAULT_ROOT
#define PREDICTOR_DEFAULT_ROOT "."
#endif
int main(int argc, char **argv) {
    std::string suffix = "2";
    fs::path root = PREDICTOR_DEFAULT_ROOT, input, output;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            auto value = [&]() {
                if (i + 1 >= argc)
                    throw std::invalid_argument("Missing value for " + argument);
                return std::string(argv[++i]);
            };
            if (argument == "--suffix")
                suffix = value();
            else if (argument == "--input")
                input = fs::u8path(value());
            else if (argument == "--output-dir")
                output = fs::u8path(value());
            else if (argument == "--root")
                root = fs::u8path(value());
            else if (argument == "--help" || argument == "-h") {
                std::cout << "Options: --suffix VALUE --input CSV --output-dir DIR --root DIR\n";
                return 0;
            } else
                throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (input.empty())
            input = root / "data" / ("pose_raw_" + suffix + ".csv");
        if (output.empty())
            output = root / "results";
        if (!fs::is_regular_file(input))
            throw std::invalid_argument("Input CSV does not exist or is not a file: " + input.string());
        run_predict(input, output, suffix);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
