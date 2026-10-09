#include "predictor_armor.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
struct Frame {
    double time;
    Eigen::Matrix<double, 11, 1> truth = Eigen::Matrix<double, 11, 1>::Zero();
    std::vector<Eigen::Vector4d> observations;
};
struct Sequence { std::string name; bool synthetic; std::vector<Frame> frames; };
struct Stats {
    int observations = 0, accepted = 0, rejected = 0, forecasts = 0, steps = 0, truth_frames = 0;
    std::array<double, 4> error{};
    double center_step = 0, w_step = 0, center_truth = 0, plate_truth = 0, horizon = 0;
};
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Expected sequence and candidate files");
        std::ifstream input(argv[1]), candidates(argv[2]);
        int count;
        input >> count;
        std::vector<Sequence> sequences(count);
        for (auto &s : sequences) {
            int length;
            input >> s.name >> s.synthetic >> length;
            s.frames.resize(length);
            for (auto &f : s.frames) {
                int n;
                input >> f.time >> n;
                if (s.synthetic) {
                    for (int i = 0; i < 11; ++i) input >> f.truth(i);
                }
                for (int k = 0; k < n; ++k) {
                    Eigen::Vector4d z = Eigen::Vector4d::Zero();
                    for (int j = 0; j < 4; ++j) input >> z(j);
                    f.observations.push_back(z);
                }
            }
        }
        if (!input) throw std::runtime_error("Invalid replay file");
        std::cout << "candidate,sequence,split,observations,accepted,rejected,forecast_count,yaw_rmse,pitch_rmse,distance_rmse,plate_yaw_rmse,center_step_rms,w_step_rms,center_truth_rmse,plate_truth_rmse,horizon_ms\n";
        std::cout << std::setprecision(12);
        std::string name;
        double v0, v1, v2, v3;
        while (candidates >> name >> v0 >> v1 >> v2 >> v3) {
            for (const auto &s : sequences) {
                ArmorEKF b{};
                b.R = Eigen::Vector4d(v0, v1, v2, v3).asDiagonal();
                std::array<Stats, 2> stats{};
                std::optional<Eigen::Matrix<double, 11, 1>> previous;
                double last = 0;
                for (size_t i = 0; i < s.frames.size(); ++i) {
                    const auto &f = s.frames[i];
                    if (!b.is_initialized) {
                        for (const auto &o : f.observations)
                            { b.initialize(o); break; }
                        last = f.time;
                        continue;
                    }
                    b.predict(f.time - last);
                    last = f.time;
                    const auto result = b.update_multi(f.observations);
                    auto &m = stats[i < s.frames.size() * 7 / 10 ? 0 : 1];
                    m.observations += static_cast<int>(f.observations.size());
                    m.accepted += static_cast<int>(result.size());
                    m.rejected += static_cast<int>(f.observations.size() - result.size());
                    if (i > 30) {
                        if (previous) {
                            double distance = 0;
                            for (int axis : {0, 2, 4}) distance += std::pow(b.X(axis) - (*previous)(axis), 2);
                            m.center_step += distance;
                            m.w_step += std::pow(b.X(7) - (*previous)(7), 2);
                            ++m.steps;
                        }
                        size_t future = i + 1;
                        while (future < s.frames.size() && s.frames[future].time < f.time + .05 - 1e-8) ++future;
                        if (future < s.frames.size()) {
                            const auto &target = s.frames[future];
                            auto forecast = b.forecast(target.time - f.time);
                            for (const auto &o : target.observations) {
                                Eigen::Vector4d residual;
                                double smallest = std::numeric_limits<double>::infinity();
                                for (int id = 0; id < 4; ++id) {
                                    auto r = armor_angular_residual(o, armor_h(forecast.state, id));
                                    if (std::abs(r(3)) < smallest) { smallest = std::abs(r(3)); residual = r; }
                                }
                                for (int k = 0; k < 4; ++k) m.error[k] += residual(k) * residual(k);
                                m.horizon += (target.time - f.time) * 1000;
                                ++m.forecasts;
                            }
                            if (s.synthetic) {
                                ++m.truth_frames;
                                const auto &left = s.frames[future - 1];
                                double fraction = (f.time + .05 - left.time) / (target.time - left.time);
                                const Eigen::Matrix<double, 11, 1> truth = left.truth + fraction * (target.truth - left.truth);
                                const auto lead = b.forecast(.05);
                                for (int axis : {0, 2, 4}) m.center_truth += std::pow(lead.state(axis) - truth(axis), 2);
                                double error = 0;
                                for (int id = 0; id < 4; ++id) {
                                    auto z = armor_h(truth, id);
                                    double smallest = std::numeric_limits<double>::infinity();
                                    double matched_error = 0;
                                    for (int other = 0; other < 4; ++other) {
                                        auto predicted = armor_h(lead.state, other);
                                        auto r = armor_angular_residual(z, predicted);
                                        if (std::abs(r(3)) < smallest) {
                                            smallest = std::abs(r(3));
                                            const double r_true = truth(8) + (id % 2 ? truth(9) : 0);
                                            const double yaw = truth(6) + id * armor_pi / 2;
                                            Eigen::Vector3d p(truth(0) + r_true * std::sin(yaw),
                                                              truth(2) + (id % 2 ? truth(10) : 0),
                                                              truth(4) - r_true * std::cos(yaw));
                                            matched_error = (lead.plates.row(other).head(3).transpose() - p).squaredNorm();
                                        }
                                    }
                                    error += matched_error / 4;
                                }
                                m.plate_truth += error;
                            }
                        }
                    }
                    previous = b.X;
                }
                for (int split = 0; split < 2; ++split) {
                    const auto &m = stats[split];
                    auto rms = [](double sum, int n) { return n ? std::sqrt(sum / n) : 0.; };
                    std::cout << name << ',' << s.name << ',' << (split ? "validation" : "training") << ','
                              << m.observations << ',' << m.accepted << ',' << m.rejected << ',' << m.forecasts;
                    for (double e : m.error) std::cout << ',' << rms(e, m.forecasts);
                    std::cout << ',' << rms(m.center_step, m.steps) << ',' << rms(m.w_step, m.steps)
                              << ',' << rms(m.center_truth, m.truth_frames) << ',' << rms(m.plate_truth, m.truth_frames)
                              << ',' << (m.forecasts ? m.horizon / m.forecasts : 0) << '\n';
                }
            }
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
