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
        if (argc != 4) throw std::runtime_error("Expected video, calibration profile or pixels, output CSV");
        camera input;
        input.open(VIDEO, argv[1]);
        if (std::string(argv[2]) == "pixels") {
            std::ofstream out(argv[3]);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << "frame_id,timestamp,count,index,center_x,center_y,score,x0,y0,x1,y1,x2,y2,x3,y3\n" << std::setprecision(17);
            do {
                const auto detections = detector::detect(input.image, ENEMY_RED);
                if (detections.empty())
                    out << input.frame_id << ',' << input.timestamp_ms << ",0,-1,,,,,,,,,,,\n";
                for (std::size_t i = 0; i < detections.size(); ++i) {
                    const auto &plate = detections[i];
                    out << input.frame_id << ',' << input.timestamp_ms << ',' << detections.size() << ',' << i
                        << ',' << plate.center.x << ',' << plate.center.y << ',' << plate.detection_score;
                    for (const auto &point : plate.vertices) out << ',' << point.x << ',' << point.y;
                    out << '\n';
                }
            } while (input.next());
            return 0;
        }
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
                if (!std::isfinite(z(2)) || observation(2) < z(2)) {
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
