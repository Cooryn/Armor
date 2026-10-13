#include "predictor_single.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

// Offline Q/R evaluation using the production single-plate update_frame path.
struct SingleNoiseFrame
{
    double timestamp_ms;
    Eigen::Vector4d truth;
    std::vector<Eigen::Vector4d> observations;
};
struct SingleNoiseSequence
{
    std::string name;
    bool truth_available;
    std::vector<SingleNoiseFrame> frames;
};

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3 && argc != 4)
            throw std::invalid_argument("Expected sequence file, candidate file and optional detailed CSV");
        std::ifstream input(argv[1]), candidates(argv[2]);
        input.exceptions(std::ios::badbit | std::ios::failbit);
        int count;
        input >> count;
        std::vector<SingleNoiseSequence> sequences(count);
        for (auto &sequence : sequences)
        {
            int length;
            input >> sequence.name >> sequence.truth_available >> length;
            sequence.frames.resize(length);
            for (auto &frame : sequence.frames)
            {
                int observations;
                input >> frame.timestamp_ms >> observations;
                if (sequence.truth_available)
                    for (int i = 0; i < 4; ++i)
                        input >> frame.truth(i);
                frame.observations.resize(observations);
                for (auto &observation : frame.observations)
                    for (int i = 0; i < 4; ++i)
                        input >> observation(i);
            }
        }
        if (!input || !candidates)
            throw std::runtime_error("Cannot read evaluation inputs");
        std::ofstream details;
        if (argc == 4)
        {
            details.open(argv[3]);
            details.exceptions(std::ios::badbit | std::ios::failbit);
            details << std::setprecision(17);
            details << "candidate,sequence,timestamp,x,vx,y,vy,z,vz,yaw,yaw_rate,future_x,future_y,future_z,future_yaw\n";
        }
        std::cout << std::setprecision(12);
        std::cout << "candidate,sequence,samples,current_rmse_m,future_rmse_m,current_yaw_rmse_rad,future_yaw_rmse_rad,"
                     "position_step_rms_m,future_step_rms_m,yaw_step_rms_rad,future_yaw_step_rms_rad,"
                     "velocity_step_rms_mps,yaw_rate_step_rms_radps,max_position_error_m\n";
        std::string name;
        double q_pos, q_yaw, yaw_variance, pitch_variance, range_variance, orientation_variance;
        while (candidates >> name >> q_pos >> q_yaw >> yaw_variance >> pitch_variance >> range_variance >> orientation_variance)
        {
            for (const auto &sequence : sequences)
            {
                predictor_single filter;
                filter.q_pos = q_pos;
                filter.q_yaw = q_yaw;
                filter.R = Eigen::Vector4d(yaw_variance, pitch_variance, range_variance, orientation_variance).asDiagonal();
                const double start_ms = sequence.frames.front().timestamp_ms;
                std::array<double, 10> sums{};
                int samples = 0, truth_samples = 0;
                double max_error = 0;
                std::optional<Eigen::Matrix<double, 8, 1>> previous;
                std::optional<Eigen::Vector4d> previous_future;
                for (int i = 0; i < static_cast<int>(sequence.frames.size()); ++i)
                {
                    const auto &frame = sequence.frames[i];
                    const auto prediction = filter.update_frame(frame.timestamp_ms, frame.observations, 100);
                    if (!filter.is_initialized)
                        continue;
                    const auto &current = prediction.current.plates[0];
                    const auto &future = prediction.future.plates[0];
                    if (details.is_open())
                    {
                        details << name << ',' << sequence.name << ',' << frame.timestamp_ms;
                        for (double value : filter.X)
                            details << ',' << value;
                        for (double value : future)
                            details << ',' << value;
                        details << '\n';
                    }
                    if (frame.timestamp_ms - start_ms >= 1000 && previous)
                    {
                        for (int axis : {0, 2, 4})
                        {
                            sums[4] += std::pow(filter.X(axis) - (*previous)(axis), 2);
                            sums[8] += std::pow(filter.X(axis + 1) - (*previous)(axis + 1), 2);
                        }
                        sums[5] += (future.head<3>() - previous_future->head<3>()).squaredNorm();
                        sums[6] += std::pow(predictor_single::wrap_to_pi(filter.X(6) - (*previous)(6)), 2);
                        sums[7] += std::pow(predictor_single::wrap_to_pi(future(3) - (*previous_future)(3)), 2);
                        sums[9] += std::pow(filter.X(7) - (*previous)(7), 2);
                        ++samples;
                        if (sequence.truth_available && i + 1 < static_cast<int>(sequence.frames.size()))
                        {
                            auto right = std::lower_bound(sequence.frames.begin() + i + 1, sequence.frames.end(),
                                                          frame.timestamp_ms + 100,
                                                          [](const auto &item, double time) { return item.timestamp_ms < time; });
                            if (right != sequence.frames.end())
                            {
                                const auto &left = *(right - 1);
                                const double fraction = (frame.timestamp_ms + 100 - left.timestamp_ms) /
                                                        (right->timestamp_ms - left.timestamp_ms);
                                Eigen::Vector4d truth = left.truth + fraction * (right->truth - left.truth);
                                truth(3) = predictor_single::wrap_to_pi(left.truth(3) + fraction *
                                    predictor_single::wrap_to_pi(right->truth(3) - left.truth(3)));
                                const double current_error = (current.head<3>() - frame.truth.head<3>()).norm();
                                sums[0] += current_error * current_error;
                                sums[1] += (future.head<3>() - truth.head<3>()).squaredNorm();
                                sums[2] += std::pow(predictor_single::wrap_to_pi(current(3) - frame.truth(3)), 2);
                                sums[3] += std::pow(predictor_single::wrap_to_pi(future(3) - truth(3)), 2);
                                max_error = std::max(max_error, current_error);
                                ++truth_samples;
                            }
                        }
                    }
                    previous = filter.X;
                    previous_future = future;
                }
                std::cout << name << ',' << sequence.name << ',' << samples;
                for (int metric = 0; metric < 10; ++metric)
                {
                    const int n = metric < 4 ? truth_samples : samples;
                    std::cout << ',' << (n ? std::sqrt(sums[metric] / n) : 0);
                }
                std::cout << ',' << max_error << '\n';
            }
        }
        if (!candidates.eof())
            throw std::runtime_error("Incomplete candidate parameters");
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
