#include "camera_frames.hpp"
#include <iomanip>
#include <iostream>

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
void convert(const fs::path &input, const fs::path &output, const CameraFrames &frames) {
    const auto csv = read_csv(input);
    std::map<std::string, size_t> columns;
    for (size_t i = 0; i < csv[0].size(); ++i)
        if (!columns.emplace(csv[0][i], i).second) throw std::invalid_argument("Duplicate input column");
    for (const char *key : {"frame_id", "timestamp", "x", "y", "z", "rvec_x", "rvec_y", "rvec_z", "coordinate_frame"})
        if (!columns.count(key)) throw std::invalid_argument(std::string("Missing raw camera pose column: ") + key);
    std::ostringstream content;
    content << "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,"
        "detection_score,reprojection_error,pnp_candidate_count,pnp_used_temporal,"
        "rvec_x,rvec_y,rvec_z,qw,qx,qy,qz,coordinate_frame,base_reference_timestamp_ms,"
        "base_origin_x_m,base_origin_y_m,base_origin_z_m\n" << std::setprecision(17);
    double last_frame = -1, last_time = -1;
    for (size_t i = 1; i < csv.size(); ++i) {
        const auto &row = csv[i];
        if (row.size() != csv[0].size() || row.at(columns.at("coordinate_frame")) != "camera")
            throw std::invalid_argument("Expected a camera-frame raw pose CSV with consistent row width");
        auto get = [&](const char *key) { return numeric(row.at(columns.at(key))); };
        const double frame = get("frame_id"), timestamp = get("timestamp");
        if (!std::isfinite(frame) || frame < 0 || std::floor(frame) != frame || frame < last_frame ||
            !std::isfinite(timestamp) || timestamp < 0 ||
            (frame == last_frame ? timestamp != last_time : timestamp <= last_time))
            throw std::invalid_argument("Invalid frame/time order in camera observations");
        last_frame = frame; last_time = timestamp;
        const Eigen::Vector3d camera_position(get("x"), get("y"), get("z"));
        const Eigen::Vector3d camera_rvec(get("rvec_x"), get("rvec_y"), get("rvec_z"));
        if (!camera_position.allFinite() || camera_position.z() <= 0 || !camera_rvec.allFinite())
            throw std::invalid_argument("Invalid PnP camera pose");
        const auto T_BC = frames.at(timestamp);
        const Eigen::Vector3d position = T_BC * camera_position;
        Eigen::Matrix3d R_CA = Eigen::Matrix3d::Identity();
        if (camera_rvec.norm() > 1e-15)
            R_CA = Eigen::AngleAxisd(camera_rvec.norm(), camera_rvec.normalized()).toRotationMatrix();
        const Eigen::Matrix3d R_BA = T_BC.linear() * R_CA;
        Eigen::Quaterniond q(R_BA);
        q.normalize();
        if (q.w() < 0) q.coeffs() *= -1;
        const Eigen::AngleAxisd aa(R_BA);
        const Eigen::Vector3d rvec = aa.axis() * aa.angle();
        const Eigen::Vector3d normal = R_BA.col(2);
        if (std::hypot(normal.x(), normal.z()) < 1e-6 || position.norm() < 1e-6)
            throw std::invalid_argument("Pose has no usable horizontal heading or distance in base frame");
        content << frame << ',' << timestamp << ',' << position.x() << ',' << position.y() << ','
            << position.z() << ',' << std::atan2(position.x(), position.z()) << ','
            << std::atan2(position.y(), std::hypot(position.x(), position.z())) << ',' << position.norm()
            << ',' << std::atan2(-normal.x(), normal.z());
        for (const char *key : {"detection_score", "reprojection_error", "pnp_candidate_count", "pnp_used_temporal"}) {
            content << ',';
            auto c = columns.find(key);
            if (c != columns.end()) {
                const double value = numeric(row.at(c->second));
                if (std::isfinite(value)) content << value;
            }
        }
        content << ',' << rvec.x() << ',' << rvec.y() << ',' << rvec.z() << ','
            << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ",base,"
            << frames.reference_time_ms() << ',' << frames.origin_in_mechanical().x() << ','
            << frames.origin_in_mechanical().y() << ',' << frames.origin_in_mechanical().z() << '\n';
    }
    if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
    std::ofstream out(output);
    out << content.str();
    out.close();
    if (!out) throw std::runtime_error("Cannot write base-frame poses");
    std::cout << "Saved " << csv.size()-1 << " base-frame poses: " << output.string() << '\n';
}
} // namespace
#ifndef CAMERA_DEFAULT_ROOT
#define CAMERA_DEFAULT_ROOT "."
#endif
int main(int argc, char **argv) {
    try {
        fs::path root = CAMERA_DEFAULT_ROOT, input, output, telemetry, calibration;
        std::string suffix = "1";
        double max_gap_ms = 100;
        std::optional<double> reference_time_ms;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() {
                if (++i >= argc) throw std::invalid_argument("Missing value for " + a);
                return std::string(argv[i]);
            };
            if (a == "--input") input = fs::u8path(value());
            else if (a == "--output") output = fs::u8path(value());
            else if (a == "--root") root = fs::u8path(value());
            else if (a == "--suffix") suffix = value();
            else if (a == "--telemetry") telemetry = fs::u8path(value());
            else if (a == "--calibration") calibration = fs::u8path(value());
            else if (a == "--sync-gap-ms") max_gap_ms = numeric(value());
            else if (a == "--origin-time-ms") reference_time_ms = numeric(value());
            else if (a == "--help" || a == "-h") {
                std::cout << "Convert full PnP poses from camera to fixed gimbal base coordinates.\n"
                    "--telemetry CSV --calibration CSV [--suffix N --root DIR --input CSV --output CSV --sync-gap-ms 100]\n"
                    "--origin-time-ms MS: fixed origin at that camera center; default is first telemetry timestamp.\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: " + a);
        }
        if (telemetry.empty() || calibration.empty())
            throw std::invalid_argument("Explicit --telemetry and --calibration are required; no assumed zero angles");
        if (input.empty()) input = root/"data"/("pose_raw_"+suffix+".csv");
        if (output.empty()) output = root/"data"/("pose_base_"+suffix+".csv");
        for (const auto &source : {input, telemetry, calibration})
            if (fs::weakly_canonical(source) == fs::weakly_canonical(output) ||
                (fs::exists(output) && fs::equivalent(source, output)))
                throw std::invalid_argument("Output must not overwrite any input file");
        convert(input, output, CameraFrames::load(telemetry, calibration, max_gap_ms, reference_time_ms));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
