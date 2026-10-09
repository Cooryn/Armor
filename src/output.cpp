#include "output.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <stdexcept>

void Output::open(const std::filesystem::path &root, const Camera &input, PredictorType model, bool preview)
{
    type_ = model;
    preview_ = preview;
    processed_ = 0;
    output_path_ = root / "results" / (input.stem_ + ".avi");
    raw_path_ = root / "data" / ("pose_raw_" + input.suffix_ + ".csv");
    if (!input.live() && std::filesystem::weakly_canonical(input.path_) == std::filesystem::weakly_canonical(output_path_))
        throw std::invalid_argument("Output video must not overwrite the input video");
    predictions_ = PredictionOutput(model, root / "results", input.suffix_, true);
    writer_.open(output_path_.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), input.fps_, input.image_.size());
    if (!writer_.isOpened())
        throw std::runtime_error("Cannot write video: " + output_path_.string());
    std::filesystem::create_directories(root / "data");
    raw_.open(raw_path_);
    base_.open(root / "data" / ("pose_base_" + input.suffix_ + ".csv"));
    transforms_.open(root / "data" / ("camera_pose_" + input.suffix_ + ".csv"));
    controls_.open(root / "results" / ("control_target_" + input.suffix_ + ".csv"));
    for (auto *stream : {&raw_, &base_, &transforms_, &controls_})
    {
        if (!*stream)
            throw std::runtime_error("Cannot open Armor CSV output");
        stream->exceptions(std::ios::badbit | std::ios::failbit);
        *stream << std::setprecision(17);
    }
    constexpr const char *pose_columns = "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,"
        "detection_score,reprojection_error,pnp_candidate_count,pnp_used_temporal,rvec_x,rvec_y,rvec_z";
    raw_ << pose_columns << ",coordinate_frame\n";
    base_ << pose_columns << ",qw,qx,qy,qz,coordinate_frame,base_reference_timestamp_ms,base_origin_x_m,base_origin_y_m,base_origin_z_m\n";
    transforms_ << "frame_id,timestamp,synchronized,qw,qx,qy,qz,x,y,z,base_reference_timestamp_ms,base_origin_x_m,base_origin_y_m,base_origin_z_m\n";
    controls_ << "frame_id,timestamp,yaw_rad,pitch_rad,valid,armor_id,status,position_variance\n";
    if (preview_)
    {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        cv::resizeWindow(window, 960, 720);
    }
    finished_ = false;
}

static void draw_prediction(cv::Mat &image, const PredictionGeometry &geometry, const Eigen::Isometry3d &camera_from_base,
                            const Solver &solver, bool future, PredictorType type)
{
    const std::array<cv::Scalar, 4> colors = {cv::Scalar(0, 255, 255), cv::Scalar(255, 180, 0),
                                            cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 100)};
    const auto project = [&](const Eigen::Vector3d &base_point) -> std::optional<cv::Point>
    {
        const Eigen::Vector3d p = camera_from_base * base_point;
        if (p.z() <= .1)
            return std::nullopt;
        std::vector<cv::Point2d> pixels;
        cv::projectPoints(std::vector<cv::Point3d>{{p.x(), p.y(), p.z()}}, cv::Vec3d(0, 0, 0),
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
        if (type == PREDICTOR_SINGLE_PLATE)
            continue;
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

bool Output::write(const Camera &input, const Solver &solver, const SolvedFrame &camera_poses,
                   const SolvedFrame &base_poses, const PoseBase &pose_base,
                   const VideoPrediction &prediction, const ControlTarget &command,
                   std::chrono::steady_clock::time_point frame_start)
{
    for (const auto *poses : {&camera_poses, &base_poses})
    {
        const bool base_frame = poses == &base_poses;
        auto &stream = base_frame ? base_ : raw_;
        for (std::size_t i = 0; i < poses->observations.size(); ++i)
        {
            const auto &armor = poses->armors.at(i);
            const auto &p = poses->observations[i].position;
            const auto &z = poses->observations[i].measurement;
            stream << input.frame_id_ << ',' << input.timestamp_ms << ',' << p.x() << ',' << p.y() << ',' << p.z();
            for (int j = 0; j < 4; ++j)
                stream << ',' << z(j);
            stream << ',' << armor.detection_score << ',' << armor.reprojection_error << ',' << armor.pnp_candidate_count << ',' << armor.pnp_used_temporal;
            const Eigen::Vector3d rvec(armor.rvec.at<double>(0), armor.rvec.at<double>(1), armor.rvec.at<double>(2));
            for (int j = 0; j < 3; ++j)
                stream << ',' << rvec(j);
            if (!base_frame)
                stream << ",camera\n";
            else
            {
                cv::Mat rotation;
                cv::Rodrigues(armor.rvec, rotation);
                Eigen::Quaterniond q(Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation.ptr<double>()));
                if (q.w() < 0)
                    q.coeffs() *= -1;
                stream << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ",base," << pose_base.reference_time_ms_;
                for (int j = 0; j < 3; ++j)
                    stream << ',' << pose_base.origin_in_mechanical_(j);
                stream << '\n';
            }
        }
    }
    predictions_.write(input.frame_id_, input.timestamp_ms, base_poses.observations, prediction);
    const auto base_from_camera = pose_base.at(input.timestamp_ms);
    const Eigen::Quaterniond q(base_from_camera.linear());
    const auto &p = base_from_camera.translation();
    transforms_ << input.frame_id_ << ',' << input.timestamp_ms << ",1," << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z()
                << ',' << p.x() << ',' << p.y() << ',' << p.z();
    cv::Mat canvas = input.image_.clone();
    drawArmors(canvas, camera_poses.armors);
    const auto camera_from_base = base_from_camera.inverse();
    draw_prediction(canvas, prediction.current, camera_from_base, solver, false, type_);
    draw_prediction(canvas, prediction.future, camera_from_base, solver, true, type_);
    transforms_ << ',' << pose_base.reference_time_ms_;
    for (int j = 0; j < 3; ++j)
        transforms_ << ',' << pose_base.origin_in_mechanical_(j);
    transforms_ << '\n';
    controls_ << input.frame_id_ << ',' << input.timestamp_ms << ',' << command.yaw_rad << ',' << command.pitch_rad << ','
              << command.valid << ',' << command.armor_id << ',' << command.status << ',' << prediction.position_variance << '\n';
    const char *model = type_ == PREDICTOR_SINGLE_PLATE ? "SinglePlate" : type_ == PREDICTOR_POLAR ? "Polar" : "Armor";
    drawVideoInfo(canvas, camera_poses.armors, cv::format("%s %s frame %lld %s  control %s",
        input.live() ? "LIVE" : "REPLAY", model, static_cast<long long>(input.frame_id_),
        prediction.status.c_str(), command.status.c_str()));
    writer_.write(canvas);
    ++processed_;
    if (!preview_)
        return true;
    cv::imshow(window, canvas);
    const auto deadline = frame_start + std::chrono::duration<double>(input.live() ? 0 : 1 / input.fps_);
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

void Output::finish()
{
    if (finished_)
        return;
    predictions_.finish();
    for (auto *stream : {&raw_, &base_, &transforms_, &controls_})
        stream->close();
    writer_.release();
    cv::VideoCapture exported(output_path_.string());
    if (!exported.isOpened() || exported.get(cv::CAP_PROP_FRAME_COUNT) != double(processed_))
        throw std::runtime_error("Output video write failed or frame count is incomplete: " + output_path_.string());
    if (preview_)
        cv::destroyAllWindows();
    finished_ = true;
    std::cout << "Processed " << processed_ << " frames. Video: " << output_path_.string() << '\n';
}

constexpr double missing = std::numeric_limits<double>::quiet_NaN();


void PredictionOutput::write(std::int64_t frame, double timestamp, const std::vector<VideoObservation> &obs,
                             const VideoPrediction &prediction)
{
    auto numbers = [](CsvOutput &csv, std::initializer_list<double> values)
    {
        for (double value : values)
        {
            csv.stream_ << (csv.first_ ? "" : ",");
            if (std::isfinite(value))
                csv.stream_ << value;
            csv.first_ = false;
        }
    };
    if (type == PREDICTOR_ARMOR)
    {
        for (const auto &d : prediction.diagnostics)
        {
            const auto &z = obs.at(d.observation_index).measurement;
            numbers(logs, {double(frame), double(d.observation_index)});
            logs.stream_ << "," << (d.accepted ? "True" : "False");
            numbers(logs, {double(d.armor_id), d.nis});
            logs.stream_ << "," << d.reason;
            numbers(logs, {timestamp, z(2), z(3), double(d.best_candidate_id), d.distance_residual});
            logs.stream_ << (base_coordinates ? ",base\n" : "\n");
            logs.first_ = true;
        }
    }
    if (!prediction.value_count)
        return;
    const std::size_t columns = type == PREDICTOR_SINGLE_PLATE ? 13 : type == PREDICTOR_POLAR ? 15
                                                                                              : 31;
    for (std::size_t i = 0; i < columns; ++i)
        numbers(results, {prediction.values[i]});
    if (type == PREDICTOR_ARMOR)
    {
        results.stream_ << "," << prediction.status;
        for (std::size_t i = 0; i < prediction.future.plates.size(); ++i)
        {
            const auto &p = prediction.future.plates[i];
            numbers(futures, {double(frame), timestamp, timestamp + prediction.horizon_ms, prediction.horizon_ms,
                              double(i), p(0), p(1), p(2), p(3)});
            futures.stream_ << "," << prediction.status;
            futures.stream_ << (base_coordinates ? ",base\n" : "\n");
            futures.first_ = true;
        }
    }
    results.stream_ << (base_coordinates ? ",base\n" : "\n");
    results.first_ = true;
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

void PredictionOutput::finish()
{
    if (finished)
        return;
    for (auto *csv : {&results, &logs, &futures})
        if (csv->stream_.is_open())
        {
            csv->stream_.flush();
            csv->stream_.close();
        }
    std::ofstream metrics(metrics_path);
    if (!metrics)
        throw std::runtime_error("Cannot write " + metrics_path.string());
    metrics.exceptions(std::ios::badbit | std::ios::failbit);
    if (type == PREDICTOR_ARMOR)
        metrics << "Metrics: prior residuals of accepted observations only; not ground-truth error.\n";
    metrics << std::fixed << std::setprecision(6);
    const std::array<const char *, 4> basic_names = {"RMSE_x", "RMSE_z", "RMSE_yaw", "RMSE_distance"},
                                      angular_names = {"RMSE_target_yaw", "RMSE_target_pitch", "RMSE_distance", "RMSE_armor_yaw"},
                                      basic_units = {"m", "m", "rad", "m"}, angular_units = {"rad", "rad", "m", "rad"};
    const auto &names = type == PREDICTOR_SINGLE_PLATE ? basic_names : angular_names;
    const auto &units = type == PREDICTOR_SINGLE_PLATE ? basic_units : angular_units;
    for (std::size_t i = 0; i < names.size(); ++i)
        metrics << names[i] << ": " << (error_counts[i] ? std::sqrt(squared_errors[i] / double(error_counts[i])) : missing)
                << ' ' << units[i] << '\n';
    metrics.close();
    finished = true;
}
