#include "camera_gimbal.hpp"
#include "camera_frames.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

namespace {
namespace fs = std::filesystem;
inline std::vector<std::vector<std::string>> read_csv(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read " + p.string());
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool in_quote = false;
    char c;
    while (f.get(c)) {
        if (c == '\"') {
            if (in_quote && f.peek() == '\"') {
                f.get(c);
                field += '\"';
            } else
                in_quote = !in_quote;
        } else if (!in_quote && (c == ',' || c == '\n' || c == '\r')) {
            row.push_back(field);
            field.clear();
            if (c != ',') {
                if (c == '\r' && f.peek() == '\n')
                    f.get(c);
                if (row.size() > 1 || !row[0].empty())
                    rows.push_back(row);
                row.clear();
            }
        } else
            field += c;
    }
    if (in_quote)
        throw std::invalid_argument("Unclosed CSV quote");
    if (!field.empty() || !row.empty()) {
        row.push_back(field);
        rows.push_back(row);
    }
    if (rows.empty())
        throw std::invalid_argument("No columns to parse from file");
    if (rows[0][0].compare(0, 3, "\xef\xbb\xbf") == 0)
        rows[0][0].erase(0, 3);
    return rows;
}
double numeric(const std::string &value) {
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
struct Record {
    double frame, timestamp;
    CameraGimbal::Output command;
};
void export_commands(const fs::path &input, const fs::path &output,
                     const CameraGimbal::Config &config, const std::optional<CameraFrames> &frames) {
    CameraGimbal controller(config);
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
    for (size_t i = 1; i < csv.size(); ++i) {
        const auto &row = csv[i];
        if (row.size() != csv.front().size())
            throw std::invalid_argument("CSV row width mismatch at row " + std::to_string(i + 1));
        auto get = [&](const std::string &key) { return numeric(row.at(columns.at(key))); };
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
        if (frames) {
            const double reference = get("base_reference_timestamp_ms");
            const Eigen::Vector3d origin(get("base_origin_x_m"), get("base_origin_y_m"), get("base_origin_z_m"));
            if (!std::isfinite(reference) || !origin.allFinite() ||
                std::abs(reference - frames->reference_time_ms()) > 1e-6 ||
                (origin - frames->origin_in_mechanical()).norm() > 1e-9)
                throw std::invalid_argument("Base origin mismatch: use the same --origin-time-ms as pose_base");
            center = frames->at(timestamp).inverse() * center;
        }
        auto command = controller.update(timestamp, center,
                                          status == "updated");
        records.push_back({frame, timestamp, command});
        previous_frame = frame;
    }
    // Validate the whole input before replacing an existing command CSV.
    if (!output.parent_path().empty())
        fs::create_directories(output.parent_path());
    std::ofstream out(output);
    if (!out)
        throw std::runtime_error("Cannot write output: " + output.string());
    out << "frame_id,timestamp,yaw_error_deg,pitch_error_deg,yaw_offset_deg,pitch_offset_deg,"
           "yaw_rate_dps,pitch_rate_dps,target_valid,control_valid,angle_limited,speed_limited,"
           "acceleration_limited,status\n" << std::setprecision(17) << std::boolalpha;
    for (const auto &r : records) {
        const auto &c = r.command;
        out << r.frame << ',' << r.timestamp << ',';
        if (std::isfinite(c.yaw_error_deg)) out << c.yaw_error_deg;
        out << ',';
        if (std::isfinite(c.pitch_error_deg)) out << c.pitch_error_deg;
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
} // namespace

#ifndef CAMERA_DEFAULT_ROOT
#define CAMERA_DEFAULT_ROOT "."
#endif
int main(int argc, char **argv) {
    try {
        CameraGimbal::Config config;
        fs::path root = CAMERA_DEFAULT_ROOT, input, output, telemetry, calibration;
        double sync_gap_ms = 100;
        std::optional<double> reference_time_ms;
        std::string suffix = "1";
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            auto value = [&]() {
                if (++i >= argc) throw std::invalid_argument("Missing value for " + argument);
                return std::string(argv[i]);
            };
            if (argument == "--suffix") suffix = value();
            else if (argument == "--root") root = fs::u8path(value());
            else if (argument == "--input") input = fs::u8path(value());
            else if (argument == "--output") output = fs::u8path(value());
            else if (argument == "--telemetry") telemetry = fs::u8path(value());
            else if (argument == "--calibration") calibration = fs::u8path(value());
            else if (argument == "--sync-gap-ms") sync_gap_ms = numeric(value());
            else if (argument == "--origin-time-ms") reference_time_ms = numeric(value());
            else if (argument == "--gain") config.gain = numeric(value());
            else if (argument == "--yaw-limit-deg") config.yaw_limit_deg = numeric(value());
            else if (argument == "--pitch-limit-deg") config.pitch_limit_deg = numeric(value());
            else if (argument == "--yaw-speed-dps") config.yaw_speed_dps = numeric(value());
            else if (argument == "--pitch-speed-dps") config.pitch_speed_dps = numeric(value());
            else if (argument == "--acceleration-dps2") config.acceleration_dps2 = numeric(value());
            else if (argument == "--deadband-deg") config.deadband_deg = numeric(value());
            else if (argument == "--max-gap-ms") config.max_gap_ms = numeric(value());
            else if (argument == "--help" || argument == "-h") {
                std::cout << "Camera-only tracking CSV exporter. Angles: right-positive yaw, up-positive pitch.\n"
                    "--suffix N --root DIR --input CSV --output CSV\n"
                    "Base-frame input: --telemetry CSV --calibration CSV --sync-gap-ms 100 [--origin-time-ms MS]\n"
                    "--gain 2 --yaw-limit-deg 45 --pitch-limit-deg 30\n"
                    "--yaw-speed-dps 60 --pitch-speed-dps 45 --acceleration-dps2 180\n"
                    "--deadband-deg 0.2 --max-gap-ms 200\n"
                    "Offsets are relative to the current optical axis, not absolute encoder angles.\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (input.empty()) input = root / "results" / ("armor_prediction_result_" + suffix + ".csv");
        if (output.empty()) output = root / "results" / ("camera_gimbal_" + suffix + ".csv");
        std::optional<CameraFrames> frames;
        if (telemetry.empty() != calibration.empty())
            throw std::invalid_argument("Specify both --telemetry and --calibration");
        if (reference_time_ms && telemetry.empty())
            throw std::invalid_argument("--origin-time-ms requires telemetry/calibration");
        if (!telemetry.empty()) frames = CameraFrames::load(telemetry, calibration, sync_gap_ms, reference_time_ms);
        for (const auto &source : {input, telemetry, calibration}) {
            if (source.empty()) continue;
            if (fs::weakly_canonical(source) == fs::weakly_canonical(output) ||
                (fs::exists(output) && fs::equivalent(source, output)))
                throw std::invalid_argument("Output must not overwrite any input file");
        }
        export_commands(input, output, config, frames);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
