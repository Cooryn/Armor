#include "predictor_polar.hpp"
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
inline const InputRow &closest(const std::vector<InputRow> &rows) {
    return *std::min_element(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
        double x = number(a, "distance"), y = number(b, "distance");
        return !std::isnan(x) && (std::isnan(y) || x < y);
    });
}
inline Eigen::MatrixXd observation(const InputRow &r) {
    return PolarEKF::make_vector({number(r, "target_yaw"), number(r, "target_pitch"),
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

std::optional<Table> run_predict_polar(const fs::path &p, const fs::path &out,
                                       const std::string &suffix) {
    if (!prepare(p, out))
        return std::nullopt;
    auto data = load(p);
    const bool base_frame = input_is_base(data);
    auto groups = group_rows(data);
    PolarEKF b;
    Table results;
    std::optional<double> last;
    for (const auto &[id, g] : groups) {
        const auto &r = closest(g);
        Eigen::MatrixXd z = observation(r);
        double ts = number(r, "timestamp"), dt = last ? (ts - *last) / 1000 : 1 / 30.;
        if (dt <= 0)
            dt = 1 / 30.;
        last = ts;
        if (!b.is_initialized) {
            b.X = PolarEKF::make_vector({number(r, "x") + .26 * std::sin(z(3)), 0, number(r, "y"), 0,
                                  number(r, "z") + .26 * std::cos(z(3)), 0, z(3), 0, .26});
            b.is_initialized = true;
            continue;
        }
        b.predict(dt);
        int aid = PolarEKF::round_even(PolarEKF::wrap_to_pi(z(3) - b.X(6)) / (PolarEKF::pi / 2));
        Eigen::MatrixXd e = PolarEKF::angular_residual(b.h(b.X, aid), z);
        b.update(z);
        results.append({{"frame_id", id},
                        {"xc", b.X(0)},
                        {"vxc", b.X(1)},
                        {"yc", b.X(2)},
                        {"vyc", b.X(3)},
                        {"zc", b.X(4)},
                        {"vzc", b.X(5)},
                        {"body_yaw", b.X(6)},
                        {"w", b.X(7)},
                        {"r", b.X(8)},
                        {"err_target_yaw", e(0)},
                        {"err_target_pitch", e(1)},
                        {"err_distance", e(2)},
                        {"err_armor_yaw", e(3)},
                        {"obs_armor_yaw", z(3)}});
    }
    if (results.rows.empty()) {
        empty_message();
        return std::nullopt;
    }
    auto csv = out / ("polar_prediction_result_" + suffix + ".csv"),
         txt = out / ("polar_rmse_result_" + suffix + ".txt");
    if (base_frame) mark_base(results, data.front());
    results.to_csv(csv);
    std::cout << "生成预测数据: " << csv.string() << '\n';
    metrics(results, txt, {"err_target_yaw", "err_target_pitch", "err_distance", "err_armor_yaw"},
            {"RMSE_target_yaw", "RMSE_target_pitch", "RMSE_distance", "RMSE_armor_yaw"},
            {"rad", "rad", "m", "rad"});
    std::cout << "生成误差统计: " << txt.string() << '\n';
    return results;
}
} // namespace predictor

#ifndef PREDICTOR_DEFAULT_ROOT
#define PREDICTOR_DEFAULT_ROOT "."
#endif
int main(int argc, char **argv) {
    using namespace predictor;
    namespace fs = std::filesystem;
    std::string suffix = "2";
    fs::path root = PREDICTOR_DEFAULT_ROOT, input, output;
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
                (void)0; // Accepted for CLI compatibility; only Armor uses adaptive noise.
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
        predictor::run_predict_polar(input, output, suffix);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
