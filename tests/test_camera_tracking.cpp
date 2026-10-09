#include "output.hpp"
#include "predictor.hpp"
#include "predictor_polar.hpp"
#include "predictor_armor.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void check(bool value, const char *message)
{
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char **argv)
{
    try
    {
        if (argc != 2) throw std::invalid_argument("Expected test output directory");
        const std::filesystem::path root = argv[1];
        std::filesystem::create_directories(root);
        const auto fixture = root / "fixture.avi";
        cv::VideoWriter writer(fixture.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, {640, 480});
        check(writer.isOpened(), "fixture writer failed");
        for (int i = 0; i < 12; ++i) writer.write(cv::Mat::zeros(480, 640, CV_8UC3));
        writer.release();

        CameraCalibration mount;
        mount.yaw_to_pitch = {.1, .01, 0};
        mount.camera_in_pitch = {.03, -.015, .15};
        mount.camera_to_pitch = Eigen::AngleAxisd(.08, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(-.12, Eigen::Vector3d::UnitZ());
        mount.angle_zero_rad = {.15, -.1, .05};
        mount.angle_direction = {-1, 1, -1};
        PoseBase base(mount);
        base.push({100, .2f, -.15f, .03f, HOST_IDLE, 0});
        GimbalConfig limits;
        limits.yaw_min_rad = limits.pitch_min_rad = -2;
        limits.yaw_max_rad = limits.pitch_max_rad = 2;
        const Eigen::Vector3d desired(-.35, .22, base.latest_state_.roll_rad);
        const Eigen::Vector3d target = base.at_angles(desired) * Eigen::Vector3d(0, 0, 3);
        VideoPrediction prediction;
        prediction.initialized = true;
        prediction.status = "updated";
        prediction.position_variance = .01;
        prediction.future.plates.emplace_back(target.x(), target.y(), target.z(), 0);
        const auto command = solve_gimbal(prediction, base, 100, 100, limits);
        check(command.valid && std::abs(command.yaw_rad - desired.x()) < 1e-6 &&
              std::abs(command.pitch_rad - desired.y()) < 1e-6, "absolute joints must include mount/offset/zero/direction");
        const Eigen::Vector3d aligned = base.at_angles({command.yaw_rad, command.pitch_rad, desired.z()}).inverse() * target;
        check(aligned.head<2>().norm() < 1e-6 && aligned.z() > 0, "command does not optically align the camera");
        check(!solve_gimbal(prediction, base, 100, 201, limits).valid, "stale state or image must invalidate control");
        prediction.status = "prediction_only";
        check(!solve_gimbal(prediction, base, 100, 100, limits).valid, "dropout must not send a valid blind forecast");
        prediction.status = "updated";
        prediction.position_variance = 2;
        check(!solve_gimbal(prediction, base, 100, 100, limits).valid, "uncertain position must invalidate control");
        prediction.position_variance = .01;
        limits.yaw_min_rad = -.1;
        limits.yaw_max_rad = .1;
        check(!solve_gimbal(prediction, base, 100, 100, limits).valid, "absolute joint limits must be enforced");

        const cv::Mat intrinsics = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
        const std::vector<cv::Point3f> object_points = {{-.0675f, -.028f, 0}, {-.0675f, .028f, 0},
                                                       {.0675f, .028f, 0}, {.0675f, -.028f, 0}};
        for (auto type : {PREDICTOR_SINGLE_PLATE, PREDICTOR_POLAR, PREDICTOR_ARMOR})
        {
            Camera camera;
            camera.open(VIDEO, fixture);
            check(camera.frame_id_ == 0 && camera.timestamp_ms == 0, "first frame clock");
            PoseBase poses;
            Solver solver(intrinsics, distortion);
            SinglePlateEKF single_plate;
            PolarEKF polar;
            ArmorEKF armor;
            Output output;
            const auto directory = root / std::to_string(type);
            output.open(directory, camera, type, false);
            int count = 0;
            do
            {
                const double t = camera.timestamp_ms;
                poses.push({t, static_cast<float>(.02 * count), static_cast<float>(-.01 * count), 0, HOST_IDLE, 0});
                const Eigen::Vector3d world(.1, .2, 3);
                const auto camera_from_base = poses.at(t).inverse();
                const Eigen::Vector3d p = camera_from_base * world;
                const Eigen::AngleAxisd rotation(camera_from_base.linear());
                const Eigen::Vector3d r = rotation.axis() * rotation.angle();
                std::vector<cv::Point2f> pixels;
                cv::projectPoints(object_points, cv::Vec3d(r.x(), r.y(), r.z()), cv::Vec3d(p.x(), p.y(), p.z()),
                                  intrinsics, distortion, pixels);
                Armor detected;
                for (int i = 0; i < 4; ++i) detected.vertices[i] = pixels[i];
                const auto camera_poses = solver.solve_frame(count >= 5 && count <= 7 ? std::vector<Armor>{} :
                                                            std::vector<Armor>{detected}, t);
                const auto fixed = poses.convert(camera_poses, t);
                if (!fixed.observations.empty())
                {
                    check((fixed.observations[0].position - world).norm() < .003, "moving camera -> fixed world");
                    check(camera_poses.armors[0].tvec.data != fixed.armors[0].tvec.data, "conversion mutated the camera pose");
                }
                VideoPrediction view;
                if (type == PREDICTOR_SINGLE_PLATE)
                    view = single_plate.update_frame(camera.frame_id_, t, fixed.observations);
                else if (type == PREDICTOR_POLAR)
                    view = polar.update_frame(camera.frame_id_, t, fixed.observations);
                else
                    view = armor.update_frame(camera.frame_id_, t, fixed.observations);
                const auto control = solve_gimbal(view, poses, t, t);
                if (count >= 5 && count <= 7)
                    check(view.status == "prediction_only" && !control.valid, "missing image observation must hold control");
                if (type == PREDICTOR_SINGLE_PLATE && count > 1)
                    check((view.current.plates[0].head<3>() - world).norm() < .01, "EKF tracked camera motion instead of fixed coordinates");
                check(output.write(camera, solver, camera_poses, fixed, poses, view, control,
                                   std::chrono::steady_clock::now()), "headless output aborted");
                ++count;
            } while (camera.next());
            check(count == 12 && camera.frame_id_ == 11, "video frame clock or EOF mismatch");
            output.finish();
            output.finish();
            std::ifstream metadata(directory / "data/camera_pose_fixture.csv");
            std::string line;
            int rows = 0;
            while (std::getline(metadata, line)) ++rows;
            check(rows == 13, "per-frame transforms must include missing detections");
        }
        Serial closed;
        closed.close();
        GimbalState state;
        check(!closed.receive(state), "closed serial has no pending sample");
        std::cout << "Armor pipeline and absolute aiming checks passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
