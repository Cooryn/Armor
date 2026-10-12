#include "output.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <stdexcept>

const char *const output::window = "Armor - current / future prediction - Esc to stop";

std::filesystem::path output::result_directory(const std::filesystem::path &root,
                                               const ::camera &camera, PredictorType model, bool base_frame)
{
    const std::array<const char *, 2> models = {"single", "armor"};
    return root / "results" / (camera.source == VIDEO ? "video" : "camera") / models[model] /
           (camera.source == VIDEO ? camera.stem : base_frame ? "mcu"
                                                              : "offline");
}

void output::open(const std::filesystem::path &root, const ::camera &camera, PredictorType model,
                  bool preview, double horizon_ms, bool base_frame)
{
    type_ = model;
    preview_ = preview;
    processed_ = video_frames_ = 0;
    first_timestamp_ms_ = last_timestamp_ms_ = measured_start_ms_ = read_total_ms_ = 0;
    previous_canvas_.release();
    video_fps_ = camera.fps;
    const auto directory = result_directory(root, camera, model, base_frame);
    output_path_ = directory / (camera.stem + ".avi");
    if (camera.source == VIDEO && std::filesystem::weakly_canonical(camera.path) == std::filesystem::weakly_canonical(output_path_))
        throw std::invalid_argument("Output video must not overwrite the input video");
    open_predictions(model, directory, camera.suffix, horizon_ms, base_frame);
    writer_.open(output_path_.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), camera.fps, camera.image.size());
    if (!writer_.isOpened())
        throw std::runtime_error("Cannot write video: " + output_path_.string());
    raw_.open(directory / ("pose_raw_" + camera.suffix + ".csv"));
    controls_.open(directory / ("control_target_" + camera.suffix + ".csv"));
    if (base_frame)
    {
        base_.open(directory / ("pose_base_" + camera.suffix + ".csv"));
        transforms_.open(directory / ("camera_pose_" + camera.suffix + ".csv"));
    }
    for (auto *stream : {&raw_, &base_, &transforms_, &controls_})
    {
        if ((stream == &base_ || stream == &transforms_) && !base_frame)
            continue;
        if (!*stream)
            throw std::runtime_error("Cannot open Armor CSV output");
        stream->exceptions(std::ios::badbit | std::ios::failbit);
        *stream << std::setprecision(17);
    }
    const char *const pose_columns = "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,"
                                     "detection_score,reprojection_error,pnp_candidate_count,pnp_used_temporal,rvec_x,rvec_y,rvec_z";
    raw_ << pose_columns << ",coordinate_frame\n";
    controls_ << "frame_id,timestamp,yaw_rad,pitch_rad,valid,armor_id,status\n";
    if (base_frame)
    {
        base_ << pose_columns << ",qw,qx,qy,qz,coordinate_frame,base_reference_timestamp_ms,base_origin_x_m,base_origin_y_m,base_origin_z_m,base_reference_yaw_rad\n";
        transforms_ << "frame_id,timestamp,pose_timestamp_ms,qw,qx,qy,qz,x,y,z,base_reference_timestamp_ms,base_origin_x_m,base_origin_y_m,base_origin_z_m,base_reference_yaw_rad\n";
    }
    if (preview_)
    {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        cv::resizeWindow(window, 960, 720);
    }
    finished_ = false;
}

void output::draw_prediction(cv::Mat &image, const PredictionGeometry &geometry, const Eigen::Isometry3d &camera_from_base,
                             const ::solver &solver, bool future)
{
    const std::array<cv::Scalar, 4> colors = {cv::Scalar(0, 255, 255), cv::Scalar(255, 180, 0),
                                              cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 100)};
    const auto project = [&](const Eigen::Vector3d &base_point) -> std::optional<cv::Point>
    {
        const Eigen::Vector3d position = camera_from_base * base_point;
        if (position.z() <= .1)
            return std::nullopt;
        std::vector<cv::Point2d> pixels;
        cv::projectPoints(std::vector<cv::Point3d>{{position.x(), position.y(), position.z()}}, cv::Vec3d(0, 0, 0),
                          cv::Vec3d(0, 0, 0), solver.camera_matrix, solver.distort_coeffs, pixels);
        const auto &pixel = pixels.front();
        return cv::Point(cvRound(pixel.x), cvRound(pixel.y));
    };
    const auto marker = [&](const Eigen::Vector3d &point, const cv::Scalar &color)
    {
        const auto pixel = project(point);
        if (!pixel || !cv::Rect(0, 0, image.cols, image.rows).contains(*pixel))
            return;
        if (future)
            cv::drawMarker(image, *pixel, color, cv::MARKER_DIAMOND, 14, 2, cv::LINE_AA);
        else
            cv::circle(image, *pixel, 5, color, 2, cv::LINE_AA);
    };
    if (geometry.center)
        marker(*geometry.center, {255, 255, 255});
    for (std::size_t i = 0; i < geometry.plates.size(); ++i)
    {
        const auto &plate = geometry.plates[i];
        const Eigen::Vector3d center = plate.head<3>();
        const auto &color = colors[i % colors.size()];
        marker(center, color);
        const Eigen::Vector3d half_width(.0675 * std::cos(plate(3)), 0, .0675 * std::sin(plate(3))), half_height(0, .028, 0);
        const std::array<Eigen::Vector3d, 4> corners = {center - half_width - half_height, center + half_width - half_height,
                                                        center + half_width + half_height, center - half_width + half_height};
        std::array<std::optional<cv::Point>, 4> pixels;
        for (int j = 0; j < 4; ++j)
            pixels[j] = project(corners[j]);
        for (int j = 0; j < 4; ++j)
            if (pixels[j] && pixels[(j + 1) % 4])
            {
                auto a = *pixels[j], b = *pixels[(j + 1) % 4];
                if (!cv::clipLine(image.size(), a, b))
                    continue;
                if (!future)
                    cv::line(image, a, b, color, 2, cv::LINE_AA);
                else
                {
                    const double length = cv::norm(b - a);
                    if (length < 1)
                        continue;
                    const cv::Point2d delta = cv::Point2d(b - a) / length;
                    for (double start = 0; start < length; start += 12)
                        cv::line(image, cv::Point(cv::Point2d(a) + start * delta),
                                 cv::Point(cv::Point2d(a) + std::min(start + 7, length) * delta), color, 2, cv::LINE_AA);
                }
            }
    }
}

bool output::write(const ::camera &camera, const ::solver &solver, const SolvedFrame &camera_poses,
                   const PredictionResult &prediction, const ControlTarget &control,
                   std::chrono::steady_clock::time_point frame_start,
                   const SolvedFrame *base_poses, const ::pose *pose_base)
{
    for (const auto *poses : {&camera_poses, base_poses})
    {
        if (!poses)
            continue;
        const bool base_frame = poses == base_poses;
        auto &stream = base_frame ? base_ : raw_;
        for (std::size_t i = 0; i < poses->observations.size(); ++i)
        {
            const auto &armor = poses->armors.at(i);
            const auto &z = poses->observations[i];
            stream << camera.frame_id << ',' << camera.timestamp_ms;
            for (int j = 0; j < 3; ++j)
                stream << ',' << armor.tvec.at<double>(j);
            for (int j = 0; j < 4; ++j)
                stream << ',' << z(j);
            stream << ',' << armor.detection_score << ',' << armor.reprojection_error << ',' << armor.pnp_candidate_count << ',' << armor.pnp_used_temporal;
            for (int j = 0; j < 3; ++j)
                stream << ',' << armor.rvec.at<double>(j);
            if (!base_frame)
                stream << ",camera\n";
            else
            {
                cv::Mat rotation;
                cv::Rodrigues(armor.rvec, rotation);
                Eigen::Quaterniond q(Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation.ptr<double>()));
                if (q.w() < 0)
                    q.coeffs() *= -1;
                stream << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ",base," << pose_base->reference_time_ms_;
                stream << ",0,0,0," << pose_base->reference_yaw_rad_ << '\n';
            }
        }
    }
    write(camera.frame_id, camera.timestamp_ms, prediction);
    Eigen::Isometry3d camera_from_base = Eigen::Isometry3d::Identity();
    if (pose_base)
    {
        const auto &base_from_camera = pose_base->base_from_camera_;
        const Eigen::Quaterniond q(base_from_camera.linear());
        const auto &position = base_from_camera.translation();
        transforms_ << camera.frame_id << ',' << camera.timestamp_ms << ',' << pose_base->latest_state_.receive_timestamp_ms
                    << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z()
                    << ',' << position.x() << ',' << position.y() << ',' << position.z()
                    << ',' << pose_base->reference_time_ms_ << ",0,0,0," << pose_base->reference_yaw_rad_ << '\n';
        camera_from_base = base_from_camera.inverse();
    }
    cv::Mat canvas = camera.image.clone();
    detector::draw_armors(canvas, camera_poses.armors);
    draw_prediction(canvas, prediction.current, camera_from_base, solver, false);
    draw_prediction(canvas, prediction.future, camera_from_base, solver, true);
    controls_ << camera.frame_id << ',' << camera.timestamp_ms << ',' << control.yaw_rad << ',' << control.pitch_rad << ','
              << control.valid() << ',' << control.armor_id << ',' << control.status << '\n';
    const char *model = type_ == PREDICTOR_SINGLE_PLATE ? "SinglePlate" : "Armor";
    std::string info = cv::format("%s %s frame %lld %s", camera.source == CAMERA ? "LIVE" : "REPLAY", model,
                                  static_cast<long long>(camera.frame_id), prediction.status.c_str());
    if (pose_base)
        info += "  control " + control.status;
    detector::draw_video_info(canvas, camera_poses.armors, info);
    return write_canvas(camera, canvas, frame_start);
}

bool output::write_canvas(const ::camera &camera, const cv::Mat &canvas,
                          std::chrono::steady_clock::time_point frame_start)
{
    if (camera.source == CAMERA)
    {
        if (processed_ == 0)
            first_timestamp_ms_ = camera.timestamp_ms;
        else
        {
            if (camera.timestamp_ms < last_timestamp_ms_)
                throw std::runtime_error("Camera timestamps must not go backwards");
            const auto target_frame = static_cast<std::int64_t>(std::ceil(
                (camera.timestamp_ms - first_timestamp_ms_) * video_fps_ / 1000 - 1e-9));
            while (video_frames_ < target_frame)
            {
                writer_.write(previous_canvas_);
                ++video_frames_;
            }
        }
        previous_canvas_ = canvas.clone();
        last_timestamp_ms_ = camera.timestamp_ms;
        if (processed_ == 5)
            measured_start_ms_ = camera.timestamp_ms;
        if (processed_ >= 5)
            read_total_ms_ += camera.read_duration_ms;
    }
    else
    {
        writer_.write(canvas);
        ++video_frames_;
    }
    ++processed_;
    if (!preview_)
        return true;
    cv::imshow(window, canvas);
    const auto deadline = frame_start + std::chrono::duration<double>(camera.source == CAMERA ? 0 : 1 / camera.fps);
    do
    {
        const double remaining = std::chrono::duration<double, std::milli>(deadline - std::chrono::steady_clock::now()).count();
        if ((cv::waitKey(static_cast<int>(std::clamp(std::ceil(remaining), 1.0, 10.0))) & 0xff) == 27)
            return false;
        try
        {
            if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1)
                return false;
        }
        catch (const cv::Exception &error)
        {
            if (error.code == cv::Error::StsNullPtr)
                return false;
            throw;
        }
    } while (std::chrono::steady_clock::now() < deadline);
    return true;
}

void output::finish()
{
    if (!predictions_finished_)
    {
        for (auto *csv : {&results, &futures})
            if (csv->is_open())
            {
                csv->close();
            }
        std::ofstream metrics(metrics_path);
        if (!metrics)
            throw std::runtime_error("Cannot write " + metrics_path.string());
        metrics.exceptions(std::ios::badbit | std::ios::failbit);
        if (type_ == PREDICTOR_ARMOR)
            metrics << "Metrics: prior residuals of accepted observations only; not ground-truth error.\n";
        metrics << std::fixed << std::setprecision(6);
        const std::array<const char *, 4> basic_names = {"RMSE_x", "RMSE_z", "RMSE_yaw", "RMSE_distance"},
                                          angular_names = {"RMSE_target_yaw", "RMSE_target_pitch", "RMSE_distance", "RMSE_armor_yaw"},
                                          basic_units = {"m", "m", "rad", "m"}, angular_units = {"rad", "rad", "m", "rad"};
        const auto &names = type_ == PREDICTOR_SINGLE_PLATE ? basic_names : angular_names;
        const auto &units = type_ == PREDICTOR_SINGLE_PLATE ? basic_units : angular_units;
        for (std::size_t i = 0; i < names.size(); ++i)
            metrics << names[i] << ": " << (error_counts[i] ? std::sqrt(squared_errors[i] / double(error_counts[i])) : std::numeric_limits<double>::quiet_NaN())
                    << ' ' << units[i] << '\n';
        metrics.close();
        predictions_finished_ = true;
    }
    if (finished_)
        return;
    for (auto *stream : {&raw_, &base_, &transforms_, &controls_})
        if (stream->is_open())
            stream->close();
    if (!previous_canvas_.empty())
    {
        writer_.write(previous_canvas_);
        ++video_frames_;
        previous_canvas_.release();
        const double span_s = (last_timestamp_ms_ - first_timestamp_ms_) / 1000;
        std::cout << "Capture span=" << span_s << "s, video duration=" << video_frames_ / video_fps_ << "s";
        if (processed_ > 6 && last_timestamp_ms_ > measured_start_ms_)
            std::cout << ", measured pipeline_fps=" << (processed_ - 6) * 1000 / (last_timestamp_ms_ - measured_start_ms_)
                      << ", mean_read_ms=" << read_total_ms_ / (processed_ - 5);
        std::cout << '\n';
    }
    writer_.release();
    cv::VideoCapture exported(output_path_.string());
    if (!exported.isOpened() || exported.get(cv::CAP_PROP_FRAME_COUNT) != double(video_frames_))
        throw std::runtime_error("Output video write failed or frame count is incomplete: " + output_path_.string());
    if (preview_)
        cv::destroyAllWindows();
    finished_ = true;
    std::cout << "Processed " << processed_ << " frames. Video: " << output_path_.string() << '\n';
}

void output::write(std::int64_t frame_id, double timestamp_ms,
                   const PredictionResult &prediction)
{
    auto numbers = [](std::ofstream &csv, std::initializer_list<double> values, bool first = false)
    {
        for (double value : values)
        {
            csv << (first ? "" : ",");
            if (std::isfinite(value))
                csv << value;
            first = false;
        }
    };
    if (prediction.status == "waiting" || prediction.status == "initialized")
        return;
    const auto &v = prediction.values;
    const auto &errors = prediction.errors;
    numbers(results, {double(frame_id)}, true);
    if (type_ == PREDICTOR_SINGLE_PLATE)
        numbers(results, {v[0], v[1], errors(0), v[2], v[3], errors(1), v[4], v[5], errors(2), v[6], v[7], errors(3)});
    else
    {
        const auto &current = prediction.current.center.value();
        const auto &future = prediction.future.center.value();
        numbers(results, {timestamp_ms, horizon_ms_, timestamp_ms + horizon_ms_, future.x(), future.y(), future.z(),
                          v[0], current.x(), current.y(), current.z(), v[1], v[2], v[3], v[4], v[5],
                          errors(0), errors(1), errors(2), errors(3), v[6], v[7], v[8]});
        results << "," << prediction.status;
        for (std::size_t i = 0; i < prediction.future.plates.size(); ++i)
        {
            const auto &position = prediction.future.plates[i];
            numbers(futures, {double(frame_id), timestamp_ms, timestamp_ms + horizon_ms_, horizon_ms_, double(i), position(0), position(1), position(2), position(3)}, true);
            futures << "," << prediction.status;
            futures << (base_coordinates ? ",base\n" : "\n");
        }
    }
    results << (base_coordinates ? ",base\n" : "\n");

    for (int i = 0; i < 4; ++i)
    {
        const double e = prediction.errors(i);
        if (std::isfinite(e))
        {
            squared_errors[i] += e * e;
            ++error_counts[i];
        }
    }
}

output::output(PredictorType model, const std::filesystem::path &directory,
               const std::string &suffix, double horizon_ms, bool base_frame)
{
    open_predictions(model, directory, suffix, horizon_ms, base_frame);
}

void output::open_predictions(PredictorType model, const std::filesystem::path &directory,
                              const std::string &suffix, double horizon_ms, bool base_frame)
{
    const char *const basic_header =
        "frame_id,predicted_x,observed_x,error_x,predicted_z,observed_z,error_z,"
        "predicted_yaw,observed_yaw,error_yaw,predicted_distance,observed_distance,error_distance";
    const char *const armor_header =
        "frame_id,timestamp,prediction_horizon_ms,prediction_timestamp,future_xc,future_yc,future_zc,"
        "future_body_yaw,xc,yc,zc,xa,za,body_yaw,pred_armor_yaw,obs_armor_yaw,"
        "err_target_yaw,err_target_pitch,err_distance,err_armor_yaw,r,dl,dh,status";

    type_ = model;
    squared_errors.fill(0);
    error_counts.fill(0);
    predictions_finished_ = false;
    base_coordinates = base_frame;
    horizon_ms_ = horizon_ms;

    const char *prefix = type_ == PREDICTOR_ARMOR ? "armor_" : "";
    const char *header = type_ == PREDICTOR_ARMOR ? armor_header : basic_header;
    auto open_csv = [base_frame](std::ofstream &csv, const std::filesystem::path &path, const char *header)
    {
        csv.open(path);
        if (!csv)
            throw std::runtime_error("Cannot write " + path.string());
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << header << (base_frame ? ",coordinate_frame\n" : "\n") << std::setprecision(17);
    };
    std::filesystem::create_directories(directory);
    open_csv(results, directory / (std::string(prefix) + "prediction_result_" + suffix + ".csv"), header);
    metrics_path = directory / (std::string(prefix) + "rmse_result_" + suffix + ".txt");
    if (type_ == PREDICTOR_ARMOR)
    {
        open_csv(futures, directory / ("armor_future_prediction_" + suffix + ".csv"),
                 "frame_id,timestamp,prediction_timestamp,prediction_horizon_ms,armor_id,x,y,z,"
                 "armor_orientation_yaw,source_status");
    }
}
