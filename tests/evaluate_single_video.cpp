#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "predictor_single.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

// Offline only: full detector/PnP/single-plate path, with no serial or camera I/O.
int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("Expected video, calibration profile, output CSV");
        camera input;
        input.open(VIDEO, argv[1]);
        solver pnp;
        pnp.use_video_profile(std::stoi(argv[2]));
        predictor_single filter;
        std::ofstream out(argv[3]);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << "frame_id,timestamp,yaw,pitch,distance,orientation,x,vx,y,vy,z,vz,armor_yaw,yaw_rate\n" << std::setprecision(17);
        do {
            auto frame = pnp.solve_frame(detector::detect(input.image, ENEMY_RED), input.timestamp_ms);
            filter.update_frame(input.timestamp_ms, frame.observations, 50);
            Eigen::Vector3d z = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
            double orientation = std::numeric_limits<double>::quiet_NaN();
            for (const auto &observation : frame.observations)
                if (predictor_single::valid_observation(observation.head<3>()) &&
                    (!std::isfinite(z(2)) || observation(2) < z(2))) {
                    z = observation.head<3>();
                    orientation = observation(3);
                }
            out << input.frame_id << ',' << input.timestamp_ms;
            for (double v : z) out << ',' << v;
            out << ',' << orientation;
            for (double v : filter.X) out << ',' << (filter.is_initialized ? v : std::numeric_limits<double>::quiet_NaN());
            out << '\n';
        } while (input.next());
        out.close();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
