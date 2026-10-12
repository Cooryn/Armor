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
        ::predictor_armor filter{};
        long long frame;
        double time, previous_time = -1;
        while (input >> frame) {
            Eigen::Vector4d z;
            if (!(input >> time >> z(0) >> z(1) >> z(2) >> z(3)))
                throw std::runtime_error("Incomplete fixture");
            if (time <= previous_time)
                throw std::runtime_error("Invalid observation or clock");
            const auto prediction = filter.update_frame(time, {z}, 50);
            previous_time = time;
            if (prediction.status == "initialized")
                continue;
            states << frame << ',' << time;
            for (int i : {0, 2, 4, 1, 3, 5, 7}) states << ',' << filter.X(i);
            states << ',' << (prediction.status == "updated" ? 1 : 0) << ',' << prediction.status << '\n';
            for (int i = 0; i < 4; ++i) {
                const auto &p = prediction.future.plates[i];
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
