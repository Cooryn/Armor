#include "predictor_single.hpp"
#include "predictor_polar.hpp"
#include "predictor_armor.hpp"
#include "output.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
struct ReplayPredictor {
    PredictorType type;
    double horizon;
    ::predictor_single single_plate;
    ::predictor_polar polar;
    ::predictor_armor armor;
    ::output output_;
    bool finished_ = false;
};
static ReplayPredictor make_replay_predictor(PredictorType type, const std::filesystem::path &directory,
                                            const std::string &suffix, double horizon = 50) {
    return {type, horizon, ::predictor_single{}, ::predictor_polar{}, ::predictor_armor{}, ::output(type, directory, suffix, horizon), false};
}
static PredictionResult replay_predictor_update(ReplayPredictor &self, std::int64_t frame, double timestamp, const std::vector<Eigen::Vector4d> &obs) {
    if (self.finished_) throw std::logic_error("Cannot update finished replay");
    PredictionResult result;
    if (self.type == PREDICTOR_SINGLE_PLATE)
        result = self.single_plate.update_frame(timestamp, obs, self.horizon);
    else if (self.type == PREDICTOR_POLAR)
        result = self.polar.update_frame(timestamp, obs, self.horizon);
    else
        result = self.armor.update_frame(timestamp, obs, self.horizon);
    self.output_.write(frame, timestamp, result);
    return result;
}
static void replay_predictor_finish(ReplayPredictor &self) {
    self.output_.finish(); self.finished_ = true;
}

static void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<typename F> static void rejects(F action) {
    bool caught = false;
    try { action(); } catch (const std::exception &) { caught = true; }
    check(caught, "expected explicit error");
}
static Eigen::Vector4d sample(double x = 0, double z = 3, double yaw = 0) {
    return {std::atan2(x, z), std::atan2(.2, std::hypot(x, z)), std::hypot(std::hypot(x, z), .2), yaw};
}
static std::vector<std::vector<std::string>> csv(const std::filesystem::path &path) {
    std::ifstream input(path);
    check(bool(input), "missing export");
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(input, line)) {
        std::vector<std::string> fields;
        std::size_t start = 0, end;
        while ((end = line.find(',', start)) != std::string::npos) {
            fields.push_back(line.substr(start, end - start)); start = end + 1;
        }
        fields.push_back(line.substr(start)); rows.push_back(fields);
    }
    return rows;
}
static std::string contents(const std::filesystem::path &path) {
    std::ifstream input(path); return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
static void regressions(const std::filesystem::path &root) {
    ReplayPredictor physical = make_replay_predictor(PREDICTOR_POLAR, root / "polar_physical", "1");
    const double heading = predictor_polar::pi / 6;
    const auto observation = sample(.13, 3 - .26 * std::cos(heading), heading);
    for (int frame = 0; frame < 20; ++frame) {
        const auto view = replay_predictor_update(physical, frame, frame * 1000. / 30, {observation});
        check((view.current.center.value() - Eigen::Vector3d(0, .2, 3)).norm() < 1e-10,
              "Polar center disagrees with PnP orientation");
        for (int i = 0; i < 4; ++i) {
            const double yaw = heading + i * predictor_polar::pi / 2;
            const Eigen::Vector3d expected(.26 * std::sin(yaw), .2, 3 - .26 * std::cos(yaw));
            check((view.current.plates[i].head<3>() - expected).norm() < 1e-10,
                  "Polar plate geometry disagrees with physical radius");
            check((view.future.plates[i].head<3>() - expected).norm() < 1e-10,
                  "Stationary Polar forecast drifted");
        }
    }
    replay_predictor_finish(physical);
    const std::array<PredictorType, 3> models = {PREDICTOR_SINGLE_PLATE, PREDICTOR_POLAR, PREDICTOR_ARMOR};
    const std::array<const char *, 3> prefixes = {"", "polar_", "armor_"};
    for (std::size_t m = 0; m < models.size(); ++m) {
        const auto folder = root / std::to_string(m);
        const auto result = std::string(prefixes[m]) + "prediction_result_1.csv";
        const auto metric = std::string(prefixes[m]) + "rmse_result_1.txt";
        for (int frames : {0, 1, 4}) {
            ReplayPredictor empty = make_replay_predictor(models[m], folder / "empty", "1");
            for (int i = 0; i < frames; ++i) check(replay_predictor_update(empty, i, i * 40., {}).status == "waiting", "empty initialized");
            replay_predictor_finish(empty); replay_predictor_finish(empty);
            check(csv(folder / "empty" / result).size() == 1, "empty output is not headers only");
            check(contents(folder / "empty" / metric).find("nan") != std::string::npos, "empty RMSE");
            rejects([&] { replay_predictor_update(empty, 10, 1000, {}); });
        }
        ReplayPredictor one = make_replay_predictor(models[m], folder / "one", "1");
        check(replay_predictor_update(one, 0, 0, {sample()}).status == "initialized", "single frame initialization");
        replay_predictor_finish(one);
        check(csv(folder / "one" / result).size() == 1, "initialization exported state");

        ReplayPredictor zero = make_replay_predictor(models[m], folder / "zero", "1", 0);
        ReplayPredictor future = make_replay_predictor(models[m], folder / "future", "1", 200);
        for (int i = 0; i < 15; ++i) {
            const std::vector<Eigen::Vector4d> obs = i > 5 ? std::vector<Eigen::Vector4d>{} :
                                                            std::vector<Eigen::Vector4d>{sample(i * .01)};
            const double time = i * 40. + (i >= 5 ? 10 : 0);
            const auto a = replay_predictor_update(zero, i, time, obs), b = replay_predictor_update(future, i, time, obs);
            check(a.status != "waiting" && b.status != "waiting", "lost during dropout");
            check(a.current.plates.size() == (m == 0 ? 1u : 4u), "plate count");
            for (std::size_t j = 0; j < a.current.plates.size(); ++j)
                check((a.current.plates[j] - b.current.plates[j]).norm() < 1e-12, "forecast mutated current state");
            if (i > 5) check(a.status == "prediction_only", "dropout status");
            if (i == 14) check((b.current.plates[0].head<3>() - b.future.plates[0].head<3>()).norm() > 1e-5,
                               "future position did not advance");
        }
        replay_predictor_finish(zero); replay_predictor_finish(future);
        const auto rows = csv(folder / "future" / result);
        check(rows.size() == 15 && rows[1][0] == "1" && rows.back()[0] == "14", "state rows omit video tail");
        if (models[m] == PREDICTOR_ARMOR) {
            for (const auto &entry : {std::pair<const char *, double>{"zero", 0}, {"future", 200}}) {
                const auto forecasts = csv(folder / entry.first / "armor_future_prediction_1.csv");
                check(forecasts.size() == 57, "future export row count");
                for (std::size_t i = 1; i < forecasts.size(); ++i) {
                    check(std::stod(forecasts[i][3]) == entry.second &&
                          std::abs(std::stod(forecasts[i][2]) - std::stod(forecasts[i][1]) - entry.second) < 1e-10,
                          "future export must use configured horizon");
                }
            }
        }
    }
    ReplayPredictor selection = make_replay_predictor(PREDICTOR_SINGLE_PLATE, root / "selection", "1");
    auto invalid = sample(); invalid(2) = -1;
    const auto seed = replay_predictor_update(selection, 0, 0, {sample(0, 4), sample(1), sample(-1), invalid});
    check(std::abs(seed.current.plates[0](0) - 1) < 1e-12, "nearest valid tie did not select first");
    const auto view = replay_predictor_update(selection, 1, 100, {sample(1, 4), sample(1), sample(-1), invalid});
    replay_predictor_update(selection, 2, 200, {invalid}); replay_predictor_finish(selection);
    const auto rows = csv(root / "selection/prediction_result_1.csv");
    check(std::stod(rows[1][2]) == 1, "selected wrong observation");
    check(rows[2][2].empty() && rows[2][3].empty() && rows[2][8].empty(), "invalid observation exported as measurement");
    check(view.status == "updated", "valid update status");

    ::predictor_armor truth{};
    Eigen::Matrix<double, 11, 1> s; s << 0, 0, .2, 0, 3, 0, 0, 0, .26, 0, 0;
    std::vector<Eigen::Vector4d> four;
    for (int i = 0; i < 4; ++i) four.push_back(predictor_armor::h(s, i));
    ReplayPredictor joint = make_replay_predictor(PREDICTOR_ARMOR, root / "joint", "1");
    replay_predictor_update(joint, 0, 0, four); replay_predictor_update(joint, 1, 40, four); replay_predictor_finish(joint);
    const auto states = csv(root / "joint/armor_prediction_result_1.csv");
    check(states[0].size() == 24 && states[1].size() == 24 && states[1].back() == "updated",
          "compact armor CSV schema or update status changed");
    truth.initialize(four[0]);
    truth.predict(.04);
    check(truth.update_multi(four).size() == 4, "four-plate joint update removed");
    check((joint.armor.X - truth.X).norm() < 1e-12 && (joint.armor.P - truth.P).norm() < 1e-12,
          "frame update must retain all four observations");
    check(csv(root / "joint/armor_future_prediction_1.csv").size() == 5, "four future poses missing");

    std::filesystem::create_directories(root / "blocked/prediction_result_1.csv");
    rejects([&] { ReplayPredictor blocked = make_replay_predictor(PREDICTOR_SINGLE_PLATE, root / "blocked", "1"); });
}
static double number(std::istream &input) {
    std::string token;
    if (!(input >> token)) throw std::runtime_error("Incomplete replay fixture");
    std::size_t end;
    const double value = std::stod(token, &end);
    check(end == token.size(), "invalid replay number"); return value;
}
static int replay(const std::string &model, const std::filesystem::path &fixture, const std::filesystem::path &output, double horizon) {
    const auto type = model == "basic" ? PREDICTOR_SINGLE_PLATE : model == "polar" ? PREDICTOR_POLAR :
                      model == "armor" ? PREDICTOR_ARMOR : throw std::invalid_argument("Unknown replay model");
    std::ifstream input(fixture); check(bool(input), "missing replay fixture");
    ReplayPredictor predictor = make_replay_predictor(type, output, "1", horizon);
    std::ofstream geometry(output / "geometry.csv");
    geometry.exceptions(std::ios::badbit | std::ios::failbit);
    geometry << "frame_id,timestamp,initialized,status";
    for (const auto *prefix : {"current", "future"}) {
        for (const auto *axis : {"x", "y", "z"}) geometry << ',' << prefix << "_center_" << axis;
        for (int i = 0; i < 4; ++i)
            for (const auto *axis : {"x", "y", "z", "yaw"}) geometry << ',' << prefix << "_plate_" << i << '_' << axis;
    }
    geometry << '\n' << std::setprecision(17);
    std::int64_t frame;
    while (input >> frame) {
        const double timestamp = number(input);
        const auto count = static_cast<int>(number(input));
        check(count >= 0, "negative replay count");
        std::vector<Eigen::Vector4d> obs;
        for (int i = 0; i < count; ++i) {
            Eigen::Vector4d o;
            for (int j = 0; j < 3; ++j) (void)number(input);
            for (int j = 0; j < 4; ++j) o(j) = number(input);
            obs.push_back(o);
        }
        const auto view = replay_predictor_update(predictor, frame, timestamp, obs);
        geometry << frame << ',' << timestamp << ',' << (view.status != "waiting" ? "True" : "False") << ',' << view.status;
        for (const auto *g : {&view.current, &view.future}) {
            for (int i = 0; i < 3; ++i) {
                geometry << ','; if (g->center) geometry << (*g->center)(i);
            }
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    geometry << ','; if (i < static_cast<int>(g->plates.size())) geometry << g->plates[i](j);
                }
        }
        geometry << '\n';
    }
    check(input.eof(), "invalid replay frame");
    replay_predictor_finish(predictor); geometry.close(); return 0;
}
int main(int argc, char **argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "--replay")
            return replay(argv[2], std::filesystem::u8path(argv[3]), std::filesystem::u8path(argv[4]), std::stod(argv[5]));
        if (argc != 2) throw std::invalid_argument("Expected a test output directory");
        regressions(std::filesystem::u8path(argv[1]));
        std::cout << "Video predictor regressions passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
