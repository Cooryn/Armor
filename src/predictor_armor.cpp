#include "predictor_armor.hpp"
#include <filesystem>
#include <variant>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace predictor {
namespace {
namespace fs = std::filesystem;
using Cell = std::variant<double, std::string, bool>;
using Row = std::vector<std::pair<std::string, Cell>>;
// DataFrame replacement: preserves insertion order, named columns and missing cells.
struct Table {
    std::vector<std::string> columns;
    std::vector<Row> rows;
    void append(Row row);
    void to_csv(const std::filesystem::path &path) const;
};

inline void Table::append(Row row) {
    for (const auto &kv : row)
        if (std::find(columns.begin(), columns.end(), kv.first) == columns.end())
            columns.push_back(kv.first);
    rows.push_back(std::move(row));
}
inline std::string quoted(const std::string &s) {
    if (s.find_first_of(",\"\r\n") == std::string::npos)
        return s;
    std::string r = "\"";
    for (char c : s) {
        r += c;
        if (c == '\"')
            r += c;
    }
    return r + '\"';
}
inline void Table::to_csv(const fs::path &p) const {
    std::ofstream out(p);
    if (!out)
        throw std::runtime_error("Cannot write " + p.string());
    for (size_t i = 0; i < columns.size(); ++i)
        out << (i ? "," : "") << quoted(columns[i]);
    out << '\n' << std::setprecision(17);
    for (const auto &row : rows) {
        for (size_t i = 0; i < columns.size(); ++i) {
            if (i)
                out << ',';
            auto it = std::find_if(row.begin(), row.end(),
                                   [&](const auto &kv) { return kv.first == columns[i]; });
            if (it == row.end())
                continue;
            const auto &v = it->second;
            if (auto d = std::get_if<double>(&v)) {
                if (!std::isnan(*d))
                    out << *d;
            } else if (auto b = std::get_if<bool>(&v))
                out << (*b ? "True" : "False");
            else
                out << quoted(std::get<std::string>(v));
        }
        out << '\n';
    }
    if (!out)
        throw std::runtime_error("CSV write failed");
}
using InputRow = std::map<std::string, std::string>;
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
inline double number(const InputRow &r, const std::string &key, bool optional = false) {
    auto it = r.find(key);
    if (it == r.end()) {
        if (optional)
            return std::numeric_limits<double>::quiet_NaN();
        throw std::invalid_argument("Missing CSV column: " + key);
    }
    auto s = it->second;
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::numeric_limits<double>::quiet_NaN();
    s = s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
    if (s.empty() || s == "NA" || s == "N/A" || s == "NULL" || s == "null" || s == "None" ||
        s == "NaN" || s == "nan" || s == "<NA>")
        return std::numeric_limits<double>::quiet_NaN();
    try {
        size_t n;
        double v = std::stod(s, &n);
        if (n != s.size())
            throw std::invalid_argument("trailing characters");
        return v;
    } catch (...) {
        if (optional)
            return std::numeric_limits<double>::quiet_NaN();
        throw std::invalid_argument("Invalid numeric CSV value in " + key + ": " + s);
    }
}
inline std::vector<InputRow> load(const fs::path &p) {
    auto raw = read_csv(p);
    std::vector<InputRow> rows;
    for (size_t i = 1; i < raw.size(); ++i) {
        InputRow r;
        for (size_t j = 0; j < raw[0].size(); ++j)
            r[raw[0][j]] = j < raw[i].size() ? raw[i][j] : "";
        rows.push_back(r);
    }
    return rows;
}
using Groups = std::map<double, std::vector<InputRow>>;
inline Groups group_rows(const std::vector<InputRow> &rows) {
    Groups g;
    for (const auto &r : rows) {
        double id = number(r, "frame_id");
        if (!std::isnan(id))
            g[id].push_back(r);
    }
    return g;
}
inline Eigen::MatrixXd observation(const InputRow &r) {
    return ArmorEKF::make_vector({number(r, "target_yaw"), number(r, "target_pitch"),
                           number(r, "distance"), number(r, "armor_orientation_yaw")});
}
// Missing tags identify legacy camera-frame CSVs. Never mix frames in one run.
inline bool input_is_base(const std::vector<InputRow> &data) {
    std::string frame;
    for (const auto &row : data) {
        const auto it = row.find("coordinate_frame");
        const std::string current = it == row.end() ? "camera" : it->second;
        if (current != "camera" && current != "base")
            throw std::invalid_argument("Unknown input coordinate_frame");
        if (!frame.empty() && frame != current)
            throw std::invalid_argument("Mixed coordinate frames in input");
        if (current == "base") {
            for (const char *key : {"base_reference_timestamp_ms", "base_origin_x_m", "base_origin_y_m", "base_origin_z_m"}) {
                const double value = number(row, key), first = number(data.front(), key);
                if (!std::isfinite(value) || !std::isfinite(first) || value != first)
                    throw std::invalid_argument("Missing or mixed base origin metadata; regenerate base CSV");
            }
            if (number(row, "base_reference_timestamp_ms") < 0)
                throw std::invalid_argument("Invalid base reference timestamp");
        }
        frame = current;
    }
    return frame == "base";
}
inline void mark_base(Table &table, const InputRow &reference) {
    table.columns.push_back("coordinate_frame");
    for (auto &row : table.rows) row.push_back({"coordinate_frame", std::string("base")});
    for (const char *key : {"base_reference_timestamp_ms", "base_origin_x_m", "base_origin_y_m", "base_origin_z_m"}) {
        table.columns.push_back(key);
        for (auto &row : table.rows) row.push_back({key, number(reference, key)});
    }
}
inline bool prepare(const fs::path &p, const fs::path &out) {
    if (!fs::exists(p)) {
        std::cout << "错误: 找不到输入文件 " << p.string() << '\n';
        return false;
    }
    fs::create_directories(out);
    return true;
}
inline void empty_message() {
    std::cout << "No prediction exported: at least two detected frames are required.\n";
}
inline double rmse(const Table &t, const std::string &name) {
    double sum = 0;
    size_t count = 0;
    for (const auto &r : t.rows)
        for (const auto &kv : r)
            if (kv.first == name) {
                double v = std::get<double>(kv.second);
                if (!std::isnan(v)) {
                    sum += v * v;
                    ++count;
                }
            }
    return count ? std::sqrt(sum / count) : std::numeric_limits<double>::quiet_NaN();
}
inline void metrics(const Table &t, const fs::path &p, const std::vector<std::string> &errors,
                    const std::vector<std::string> &names, const std::vector<std::string> &units,
                    bool armor = false) {
    std::ofstream f(p);
    if (!f)
        throw std::runtime_error("Cannot write " + p.string());
    if (armor)
        f << "Metrics: prior residuals of accepted observations only; not ground-truth error.\n";
    f << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < errors.size(); ++i)
        f << names[i] << ": " << rmse(t, errors[i]) << ' ' << units[i] << '\n';
}
} // namespace

std::optional<Table> run_predict_armor(const fs::path &p, const fs::path &out,
                                       const std::string &suffix, bool adaptive, double horizon) {
    if (!std::isfinite(horizon) || horizon < 0)
        throw std::invalid_argument("Prediction horizon must be finite and nonnegative (ms)");
    if (!prepare(p, out))
        return std::nullopt;
    auto data = load(p);
    const bool base_frame = input_is_base(data);
    auto groups = group_rows(data);
    if (groups.size() < 2) {
        empty_message();
        return std::nullopt;
    }
    for (const auto &r : data) {
        double id = number(r, "frame_id"), ts = number(r, "timestamp");
        if (!std::isfinite(id) || !std::isfinite(ts) || id < 0 || std::floor(id) != id ||
            id >= double(std::numeric_limits<long long>::max()))
            throw std::invalid_argument(
                "Frame IDs and timestamps must be finite, valid video times");
    }
    std::optional<double> previous;
    for (const auto &[id, g] : groups) {
        (void)id;
        double ts = number(g[0], "timestamp");
        if (previous && ts <= *previous)
            throw std::invalid_argument(
                "Timestamps must increase between frames; regenerate legacy CSV");
        previous = ts;
    }
    for (const auto &[id, g] : groups) {
        (void)id;
        for (const auto &r : g)
            if (number(r, "timestamp") != number(g[0], "timestamp"))
                throw std::invalid_argument("All observations of a frame must share its timestamp");
    }
    ArmorEKF b(16, 25 * ArmorEKF::pi / 180, .5, adaptive);
    b.base_frame = base_frame;
    Table results, logs, futures;
    double last = 0;
    auto right = groups.begin();
    for (long long frame = static_cast<long long>(groups.begin()->first),
                   end = static_cast<long long>(groups.rbegin()->first);
         frame <= end; ++frame) {
        double id = static_cast<double>(frame);
        while (right->first < id)
            ++right;
        double ts = number(right->second[0], "timestamp");
        if (right->first != id) {
            auto left = std::prev(right);
            double t0 = number(left->second[0], "timestamp");
            ts = t0 + (ts - t0) / (right->first - left->first) * (id - left->first);
        }
        std::vector<InputRow> group;
        if (right->first == id)
            group = right->second;
        std::vector<Observation> obs;
        for (const auto &r : group)
            obs.push_back({observation(r), std::nullopt, number(r, "detection_score", true),
                           number(r, "reprojection_error", true)});
        if (!b.is_initialized) {
            int seed = -1;
            for (size_t i = 0; i < obs.size(); ++i)
                if (b.valid_observation(obs[i].Z_obs) &&
                    (seed < 0 || obs[i].Z_obs(2) < obs[seed].Z_obs(2)))
                    seed = static_cast<int>(i);
            if (seed < 0) {
                for (size_t i = 0; i < obs.size(); ++i)
                    logs.append({{"frame_id", id},
                                 {"observation_index", double(i)},
                                 {"accepted", false},
                                 {"armor_id", -1.},
                                 {"nis", std::numeric_limits<double>::quiet_NaN()},
                                 {"reason", std::string("invalid")}});
                continue;
            }
            b.initialize(obs[seed].Z_obs);
            last = ts;
            for (size_t i = 0; i < obs.size(); ++i) {
                bool selected = static_cast<int>(i) == seed;
                logs.append({{"frame_id", id},
                             {"observation_index", double(i)},
                             {"accepted", selected},
                             {"armor_id", selected ? 0. : -1.},
                             {"nis", std::numeric_limits<double>::quiet_NaN()},
                             {"reason",
                              std::string(selected ? "initialization" : "initialization_unused")}});
            }
            continue;
        }
        b.predict((ts - last) / 1000);
        last = ts;
        Eigen::MatrixXd prior = b.X;
        auto [matches, diagnostics] = b.update_multi(obs);
        for (const auto &d : diagnostics) {
            const auto &r = group[d.observation_index];
            Row row = {{"frame_id", id},
                       {"timestamp", ts},
                       {"observed_distance", number(r, "distance")},
                       {"observed_armor_yaw", number(r, "armor_orientation_yaw")},
                       {"observation_index", double(d.observation_index)},
                       {"accepted", d.accepted},
                       {"armor_id", double(d.armor_id)},
                       {"nis", d.nis},
                       {"reason", d.reason},
                       {"best_candidate_id", double(d.best_candidate_id)},
                       {"distance_residual", d.distance_residual}};
            if (b.valid_observation(obs[d.observation_index].Z_obs)) {
                row.push_back({"target_yaw_noise_scale", d.scales(0)});
                row.push_back({"target_pitch_noise_scale", d.scales(1)});
                row.push_back({"distance_noise_scale", d.scales(2)});
                row.push_back({"yaw_noise_scale", d.scales(3)});
            }
            logs.append(row);
        }
        int aid = 0;
        Eigen::MatrixXd pred, errors = Eigen::MatrixXd::Constant(4, 1, std::numeric_limits<double>::quiet_NaN());
        double obs_yaw = std::numeric_limits<double>::quiet_NaN();
        if (!matches.empty()) {
            const auto &m =
                *std::min_element(matches.begin(), matches.end(), [](const auto &a, const auto &c) {
                    return a.Z_obs(2) < c.Z_obs(2);
                });
            aid = m.armor_id;
            pred = m.predicted;
            errors = m.residual * (-1);
            obs_yaw = m.Z_obs(3);
        } else {
            for (int i = 1; i < 4; ++i)
                if (b.h(prior, i)(2) < b.h(prior, aid)(2))
                    aid = i;
            pred = b.h(prior, aid);
        }
        double ri = prior(8) + (aid % 2 ? prior(9) : 0);
        auto forecast = b.forecast(horizon / 1000);
        const auto &f = forecast.state;
        std::string status = matches.empty() ? "prediction_only" : "updated";
        for (int i = 0; i < 4; ++i)
            futures.append({{"frame_id", id},
                            {"timestamp", ts},
                            {"prediction_timestamp", ts + horizon},
                            {"prediction_horizon_ms", horizon},
                            {"armor_id", double(i)},
                            {"x", forecast.plates(i, 0)},
                            {"y", forecast.plates(i, 1)},
                            {"z", forecast.plates(i, 2)},
                            {"armor_orientation_yaw", forecast.plates(i, 3)},
                            {"source_status", status}});
        results.append({{"frame_id", id},
                        {"timestamp", ts},
                        {"prediction_horizon_ms", horizon},
                        {"prediction_timestamp", ts + horizon},
                        {"future_xc", f(0)},
                        {"future_yc", f(2)},
                        {"future_zc", f(4)},
                        {"future_body_yaw", f(6)},
                        {"xc", b.X(0)},
                        {"yc", b.X(2)},
                        {"zc", b.X(4)},
                        {"vxc", b.X(1)},
                        {"vyc", b.X(3)},
                        {"vzc", b.X(5)},
                        {"w", b.X(7)},
                        {"xa", prior(0) + ri * std::sin(pred(3))},
                        {"za", prior(4) - ri * std::cos(pred(3))},
                        {"armor_id", double(aid)},
                        {"body_yaw", b.X(6)},
                        {"pred_armor_yaw", pred(3)},
                        {"obs_armor_yaw", obs_yaw},
                        {"err_target_yaw", errors(0)},
                        {"err_target_pitch", errors(1)},
                        {"err_distance", errors(2)},
                        {"err_armor_yaw", errors(3)},
                        {"r", b.X(8)},
                        {"dl", b.X(9)},
                        {"dh", b.X(10)},
                        {"observation_count", double(obs.size())},
                        {"accepted_count", double(matches.size())},
                        {"rejected_count", double(obs.size() - matches.size())},
                        {"status", status}});
    }
    if (results.rows.empty()) {
        empty_message();
        return std::nullopt;
    }
    if (base_frame) {
        mark_base(results, data.front());
        mark_base(logs, data.front());
        mark_base(futures, data.front());
        futures.columns.insert(futures.columns.end(), {"qw", "qx", "qy", "qz"});
        for (auto &row : futures.rows) {
            const auto angle = std::find_if(row.begin(), row.end(),
                [](const auto &v) { return v.first == "armor_orientation_yaw"; });
            const double yaw = std::get<double>(angle->second);
            row.push_back({"qw", std::cos(yaw / 2)});
            row.push_back({"qx", 0.});
            row.push_back({"qy", -std::sin(yaw / 2)});
            row.push_back({"qz", 0.});
        }
    }
    logs.to_csv(out / ("armor_observation_diagnostics_" + suffix + ".csv"));
    results.to_csv(out / ("armor_prediction_result_" + suffix + ".csv"));
    futures.to_csv(out / ("armor_future_prediction_" + suffix + ".csv"));
    metrics(results, out / ("armor_rmse_result_" + suffix + ".txt"),
            {"err_target_yaw", "err_target_pitch", "err_distance", "err_armor_yaw"},
            {"RMSE_target_yaw", "RMSE_target_pitch", "RMSE_distance", "RMSE_armor_yaw"},
            {"rad", "rad", "m", "rad"}, true);
    return results;
}
} // namespace predictor

#ifndef PREDICTOR_DEFAULT_ROOT
#define PREDICTOR_DEFAULT_ROOT "."
#endif
int main(int argc, char **argv) {
    using namespace predictor;
    namespace fs = std::filesystem;
    std::string suffix = "1";
    fs::path root = PREDICTOR_DEFAULT_ROOT, input, output;
    bool fixed = false;
    double horizon = 50;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value = [&]() {
                if (i + 1 >= argc)
                    throw std::invalid_argument("Missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--suffix")
                suffix = value();
            else if (a == "--input")
                input = fs::u8path(value());
            else if (a == "--output-dir")
                output = fs::u8path(value());
            else if (a == "--root")
                root = fs::u8path(value());
            else if (a == "--fixed-noise")
                fixed = true;
            else if (a == "--prediction-horizon-ms") {
                std::string s = value();
                size_t n;
                horizon = std::stod(s, &n);
                if (n != s.size())
                    throw std::invalid_argument("Invalid horizon");
            } else if (a == "--help" || a == "-h") {
                std::cout << "Options: --suffix VALUE --input CSV --output-dir DIR --fixed-noise "
                             "--prediction-horizon-ms MS --root DIR\n";
                return 0;
            } else
                throw std::invalid_argument("Unknown argument: " + a);
        }
        if (!std::isfinite(horizon) || horizon < 0)
            throw std::invalid_argument("--prediction-horizon-ms must be finite and nonnegative");
        if (input.empty())
            input = root / "data" / ("pose_raw_" + suffix + ".csv");
        if (output.empty())
            output = root / "results";
        if (!fs::is_regular_file(input))
            throw std::invalid_argument("Input CSV does not exist or is not a file: " + input.string());
        predictor::run_predict_armor(input, output, suffix, !fixed, horizon);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
