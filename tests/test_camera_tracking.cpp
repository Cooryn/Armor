#include "log_output.hpp"
#include "video_output.hpp"
#include "predictor_single.hpp"
#include "predictor_armor.hpp"
#include <algorithm>
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

        for (int scenario = 0; scenario < 4; ++scenario)
        {
            std::vector<double> timestamps;
            if (scenario == 0)
                timestamps = {5000};
            else if (scenario == 1)
                for (int i = 0; i <= 200; ++i) timestamps.push_back(5000 + i * 100.);
            else if (scenario == 2)
                for (int i = 0; i <= 200; ++i) timestamps.push_back(5000 + i * 5.);
            else
                timestamps = {5000, 5005, 5100, 5400, 6100, 8400, 12500, 18000, 25000};
            ::camera camera;
            camera.source = CAMERA;
            camera.stem = "timeline";
            camera.suffix = "timeline";
            camera.fps = 60;
            camera.image = cv::Mat(120, 160, CV_8UC3);
            ::solver solver;
            ::log_output output;
            const auto directory = root / ("timeline-" + std::to_string(scenario));
            output.open(directory, camera, PREDICTOR_SINGLE_PLATE, 50, false);
            video_output recording(log_output::result_directory(directory, camera, PREDICTOR_SINGLE_PLATE, false), camera);
            for (int i = 0; i < static_cast<int>(timestamps.size()); ++i)
            {
                camera.frame_id = static_cast<std::int64_t>(i);
                camera.timestamp_ms = timestamps[i];
                camera.image.setTo(cv::Scalar::all(i % 2 ? 180 : 40));
                check(output.write(camera, {}, {}, {}), "timestamp output");
                recording.record(camera, camera.image);
            }
            recording.finish();
            recording.finish();
            output.finish();
            output.finish();
            cv::VideoCapture recorded((log_output::result_directory(directory, camera, PREDICTOR_SINGLE_PLATE, false) /
                                       "timeline.avi").string());
            check(recorded.isOpened(), "timestamp recording opens");
            const double span = (timestamps.back() - timestamps.front()) / 1000;
            const int expected_frames = static_cast<int>(std::ceil(span * 60 - 1e-9)) + 1;
            check(recorded.get(cv::CAP_PROP_FRAME_COUNT) == expected_frames, "timestamp recording frame count");
            check(std::abs(recorded.get(cv::CAP_PROP_FPS) - 60) < 1e-6, "recording FPS metadata");
            check(std::abs(expected_frames / 60. - span) <= 2. / 60 + 1e-9, "timestamp duration quantization");
            if (span >= 20)
                check(std::abs(expected_frames / 60. - span) / span < .05, "20 second recording within five percent");
            cv::Mat decoded;
            int frame = 0;
            std::size_t observation = 0;
            while (recorded.read(decoded))
            {
                const double timestamp = timestamps.front() + frame * 1000. / 60;
                while (observation + 1 < timestamps.size() && timestamps[observation + 1] <= timestamp + 1e-6)
                    ++observation;
                const int expected_color = observation % 2 ? 180 : 40;
                check(std::abs(int(decoded.at<cv::Vec3b>(115, 140)[0]) - expected_color) < 8,
                      "resampling holds the preceding frame and drops faster frames");
                ++frame;
            }
            check(frame == expected_frames, "all resampled frames decode");
        }

        CameraCalibration mount;
        mount.yaw_to_pitch = {.1, .01, 0};
        mount.camera_in_pitch = {.03, -.015, .15};
        mount.camera_to_pitch = Eigen::AngleAxisd(.08, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(-.12, Eigen::Vector3d::UnitZ());
        mount.angle_zero_rad = {.15, -.1, .05};
        mount.angle_direction = {-1, 1, -1};
        ::pose base(mount);
        base.update({100, .2f, -.15f, .03f, HOST_IDLE, 0});
        GimbalConfig limits;
        limits.yaw_min_rad = limits.pitch_min_rad = -2;
        limits.yaw_max_rad = limits.pitch_max_rad = 2;
        const Eigen::Vector3d desired(-.35, .22, base.latest_state_.roll_rad);
        const Eigen::Vector3d target = base.at_angles(desired) * Eigen::Vector3d(0, 0, 3);
        PredictionResult prediction;
        prediction.status = "updated";
        prediction.future.plates.emplace_back(target.x(), target.y(), target.z(), 0);
        const auto command = gimbal::solve(prediction, base, 100, 100, limits);
        check(command.valid() && std::abs(command.yaw_rad - desired.x()) < 1e-6 &&
              std::abs(command.pitch_rad - desired.y()) < 1e-6, "absolute joints must include mount/offset/zero/direction");
        const Eigen::Vector3d aligned = base.at_angles({command.yaw_rad, command.pitch_rad, desired.z()}).inverse() * target;
        check(aligned.head<2>().norm() < 1e-6 && aligned.z() > 0, "command does not optically align the camera");
        check(!gimbal::solve(prediction, base, 100, 201, limits).valid(), "stale state or image must invalidate control");
        prediction.status = "prediction_only";
        const Eigen::Vector3d forecast_angles(-.25, .18, desired.z());
        const Eigen::Vector3d forecast_target = base.at_angles(forecast_angles) * Eigen::Vector3d(0, 0, 3);
        prediction.future.plates[0].head<3>() = forecast_target;
        const auto forecast_control = gimbal::solve(prediction, base, 100, 100, limits);
        check(forecast_control.valid() && std::abs(forecast_control.yaw_rad - forecast_angles.x()) < 1e-6 &&
              std::abs(forecast_control.pitch_rad - forecast_angles.y()) < 1e-6,
              "dropout must track the new predicted position");
        check(!gimbal::solve(prediction, base, 100, 201, limits).valid(), "predicted tracking must still reject stale data");
        PredictionResult waiting;
        check(!gimbal::solve(waiting, base, 100, 100, limits).valid(), "uninitialized predictor cannot track");
        auto empty_forecast = prediction;
        empty_forecast.future.plates.clear();
        check(!gimbal::solve(empty_forecast, base, 100, 100, limits).valid(), "empty forecast cannot track");
        limits.yaw_min_rad = -.1;
        limits.yaw_max_rad = .1;
        check(!gimbal::solve(prediction, base, 100, 100, limits).valid(), "absolute joint limits must be enforced");

        const cv::Mat intrinsics = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
        const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
        const std::vector<cv::Point3f> object_points = {{-.0675f, -.028f, 0}, {-.0675f, .028f, 0},
                                                       {.0675f, .028f, 0}, {.0675f, -.028f, 0}};
        for (auto type : {PREDICTOR_SINGLE_PLATE, PREDICTOR_ARMOR})
        {
            ::camera camera;
            camera.open(VIDEO, fixture);
            check(camera.frame_id == 0 && camera.timestamp_ms == 0, "first frame clock");
            CameraCalibration pipeline_mount;
            pipeline_mount.yaw_to_pitch = {.1, .01, 0};
            pipeline_mount.camera_in_pitch = {.03, -.015, .15};
            ::pose poses(pipeline_mount);
            ::solver solver(intrinsics, distortion);
            ::predictor_single single_plate;
            ::predictor_armor armor;
            ::log_output output;
            const auto directory = root / std::to_string(type);
            output.open(directory, camera, type, 50);
            video_output video(log_output::result_directory(directory, camera, type, true), camera);
            int count = 0;
            do
            {
                const double t = camera.timestamp_ms;
                if (count != 2 && count != 6)
                    poses.update({std::max(0., t - 5), static_cast<float>(.35 + .02 * count),
                                  static_cast<float>(.12 - .01 * count), 0, HOST_IDLE, 0});
                const Eigen::Vector3d world(.1, .2, 3);
                const auto camera_from_base = poses.base_from_camera_.inverse();
                const Eigen::Vector3d p = camera_from_base * world;
                const Eigen::AngleAxisd rotation(camera_from_base.linear());
                const Eigen::Vector3d r = rotation.axis() * rotation.angle();
                std::vector<cv::Point2f> pixels;
                cv::projectPoints(object_points, cv::Vec3d(r.x(), r.y(), r.z()), cv::Vec3d(p.x(), p.y(), p.z()),
                                  intrinsics, distortion, pixels);
                ::armor detected;
                for (int i = 0; i < 4; ++i) detected.vertices[i] = pixels[i];
                const auto camera_poses = solver.solve_frame(count >= 5 && count <= 7 ? std::vector<::armor>{} :
                                                            std::vector<::armor>{detected}, t);
                const auto fixed = poses.convert(camera_poses);
                if (!fixed.observations.empty())
                {
                    check((Eigen::Vector3d(fixed.armors[0].tvec.at<double>(0), fixed.armors[0].tvec.at<double>(1), fixed.armors[0].tvec.at<double>(2)) - world).norm() < .003, "moving camera -> fixed world");
                    check(camera_poses.armors[0].tvec.data != fixed.armors[0].tvec.data, "conversion mutated the camera pose");
                }
                PredictionResult view;
                if (type == PREDICTOR_SINGLE_PLATE)
                    view = single_plate.update_frame(t, fixed.observations);
                else
                    view = armor.update_frame(t, fixed.observations);
                const auto control = gimbal::solve(view, poses, t, t);
                if (count >= 5 && count <= 7)
                    check(view.status == "prediction_only" && control.valid(), "missing image observation must continue predicted tracking");
                if (type == PREDICTOR_SINGLE_PLATE && count > 1)
                    check((view.current.plates[0].head<3>() - world).norm() < .01, "EKF tracked camera motion instead of fixed coordinates");
                check(output.write(camera, camera_poses, view, control, &fixed, &poses), "headless output aborted");
                check(video.write(camera, solver, camera_poses, std::chrono::steady_clock::now(), &view, type, &control, &poses), "prediction drawing and recording");
                ++count;
            } while (camera.next());
            check(count == 12 && camera.frame_id == 11, "video frame clock or EOF mismatch");
            video.finish();
            output.finish();
            output.finish();
            const auto results_directory = log_output::result_directory(directory, camera, type, true);
            std::ifstream metadata(results_directory / "camera_pose_fixture.csv");
            std::string line;
            int rows = 0;
            while (std::getline(metadata, line)) ++rows;
            check(rows == 13, "per-frame transforms must include missing detections");
        }
        for (auto source : {VIDEO, CAMERA})
        for (auto type : {PREDICTOR_SINGLE_PLATE, PREDICTOR_ARMOR})
        {
            ::camera camera;
            camera.open(VIDEO, fixture);
            // Use recorded frames to exercise live output without opening a physical camera.
            camera.source = source;
            if (source == CAMERA)
            {
                camera.stem = "camera_live";
                camera.suffix = "live";
            }
            const auto suffix = camera.suffix;
            ::solver solver(intrinsics, distortion);
            ::predictor_single single_plate;
            ::predictor_armor armor;
            ::log_output output;
            const auto directory = root / ((source == VIDEO ? "video-" : "camera-offline-") + std::to_string(type));
            output.open(directory, camera, type, 50, false);
            int count = 0;
            do
            {
                if (source == CAMERA)
                    camera.timestamp_ms += 100000;
                std::vector<cv::Point2f> pixels;
                cv::projectPoints(object_points, cv::Vec3d(.1, -.2, .05), cv::Vec3d(.2, .1, 3),
                                  intrinsics, distortion, pixels);
                ::armor detected;
                for (int i = 0; i < 4; ++i)
                    detected.vertices[i] = pixels[i];
                const auto camera_poses = solver.solve_frame({detected}, camera.timestamp_ms);
                check(camera_poses.observations.size() == 1, "camera-coordinate pose solved");
                PredictionResult replay_prediction;
                if (type == PREDICTOR_SINGLE_PLATE)
                    replay_prediction = single_plate.update_frame(camera.timestamp_ms, camera_poses.observations);
                else
                    replay_prediction = armor.update_frame(camera.timestamp_ms, camera_poses.observations);
                check(output.write(camera, camera_poses, replay_prediction, gimbal::simulate(replay_prediction)),
                      "camera-coordinate output works without a pose or control object");
                if (type == PREDICTOR_SINGLE_PLATE)
                    check((replay_prediction.current.plates[0].head<3>() - Eigen::Vector3d(.2, .1, 3)).norm() < .003,
                          "offline prediction must retain the PnP camera position");
                ++count;
                camera.source = VIDEO;
                const bool has_next = camera.next();
                camera.source = source;
                if (!has_next)
                    break;
            } while (true);
            output.finish();
            check(count == 12, "camera-coordinate processing keeps all input frames");
            const auto results_directory = log_output::result_directory(directory, camera, type, false);
            const std::array<const char *, 2> model_names = {"single", "armor"};
            check(results_directory == directory / "results" / (source == VIDEO ? "video" : "camera") /
                  model_names[type] / (source == VIDEO ? "fixture" : "offline"),
                  "outputs must be grouped by source, model and input or camera mode");
            if (source == CAMERA)
                check(log_output::result_directory(directory, camera, type, true) ==
                      directory / "results/camera" / model_names[type] / "mcu",
                      "MCU and offline camera outputs must have separate directories");
            check(!std::filesystem::exists(results_directory / ("pose_base_" + suffix + ".csv")) &&
                  !std::filesystem::exists(results_directory / ("camera_pose_" + suffix + ".csv")) &&
                  std::filesystem::exists(results_directory / ("control_target_" + suffix + ".csv")),
                  "offline output includes simulated control but no base poses");
            std::ifstream controls(results_directory / ("control_target_" + suffix + ".csv"));
            std::string control_line;
            std::getline(controls, control_line);
            check(control_line == "frame_id,timestamp,yaw_rad,pitch_rad,valid,armor_id,status", "simulation control header");
            int control_rows = 0;
            while (std::getline(controls, control_line))
                ++control_rows;
            check(control_rows == camera.frame_id + 1, "one simulated control per processed frame");
            std::ifstream raw(results_directory / ("pose_raw_" + suffix + ".csv"));
            std::string line;
            std::getline(raw, line);
            check(line == "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,rvec_x,rvec_y,rvec_z,coordinate_frame",
                  "camera pose CSV must contain only pose data");
            int rows = 0;
            while (std::getline(raw, line))
            {
                check(std::count(line.begin(), line.end(), ',') == 12, "camera pose CSV column count");
                check(line.substr(line.size() - 7) == ",camera", "offline observations stay in camera coordinates");
                ++rows;
            }
            check(rows == 12, "camera-coordinate processing exports every observation");
        }
        ::serial closed;
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
