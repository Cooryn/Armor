// Test-only bridge from typed base-frame observations to the production EKF API.
#include "predictor_armor.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Expected observation fixture and output directory");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot read fixture");
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        std::ofstream states(output / "states.csv"), futures(output / "futures.csv");
        states.exceptions(std::ios::failbit | std::ios::badbit);
        futures.exceptions(std::ios::failbit | std::ios::badbit);
        states << "frame_id,timestamp,xc,yc,zc,vxc,vyc,vzc,w,accepted_count,status\n" << std::setprecision(17);
        futures << "frame_id,timestamp,armor_id,x,y,z,qw,qx,qy,qz\n" << std::setprecision(17);
        ArmorEKF filter{};
        filter.base_frame = true;
        long long frame;
        double time, previous_time = -1;
        while (input >> frame) {
            ArmorMeasurement z;
            if (!(input >> time >> z(0) >> z(1) >> z(2) >> z(3)))
                throw std::runtime_error("Incomplete fixture");
            if (!armor_valid_observation(filter, z) || time <= previous_time)
                throw std::runtime_error("Invalid observation or clock");
            if (!filter.is_initialized) {
                armor_initialize(filter, z);
                previous_time = time;
                continue;
            }
            armor_predict(filter, (time - previous_time) / 1000);
            previous_time = time;
            const auto matches = armor_update_multi(filter, {{z}}).first;
            states << frame << ',' << time;
            for (int i : {0, 2, 4, 1, 3, 5, 7}) states << ',' << filter.X(i);
            states << ',' << matches.size() << ',' << (matches.empty() ? "prediction_only" : "updated") << '\n';
            const auto future = armor_forecast(filter, .05);
            for (int i = 0; i < 4; ++i) {
                const auto p = future.plates.row(i);
                futures << frame << ',' << time << ',' << i << ',' << p(0) << ',' << p(1) << ',' << p(2)
                        << ',' << std::cos(p(3) / 2) << ",0," << -std::sin(p(3) / 2) << ",0\n";
            }
        }
        if (!input.eof()) throw std::runtime_error("Invalid fixture frame");
        states.close(); futures.close();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
