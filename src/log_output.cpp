#include "log_output.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>


std::filesystem::path log_output::result_directory(const std::filesystem::path &root,
                                               const ::camera &camera, PredictorType model, bool base_frame)
{
    const std::array<const char *, 2> models = {"single", "armor"};
    return root / "results" / (camera.source == VIDEO ? "video" : "camera") / models[model] /
           (camera.source == VIDEO ? camera.stem : base_frame ? "mcu"
                                                              : "offline");
}

void log_output::open(const std::filesystem::path &root, const ::camera &camera, PredictorType model,
                  double horizon_ms, bool base_frame)
{
    type_ = model;
    const auto directory = result_directory(root, camera, model, base_frame);
    open_predictions(model, directory, camera.suffix, horizon_ms, base_frame);
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
                                     "rvec_x,rvec_y,rvec_z";
    raw_ << pose_columns << ",coordinate_frame\n";
    controls_ << "frame_id,timestamp,yaw_rad,pitch_rad,valid,armor_id,status\n";
    if (base_frame)
    {
        base_ << pose_columns << ",qw,qx,qy,qz,coordinate_frame,base_reference_timestamp_ms,base_origin_x_m,base_origin_y_m,base_origin_z_m,base_reference_yaw_rad\n";
        transforms_ << "frame_id,timestamp,pose_timestamp_ms,qw,qx,qy,qz,x,y,z,base_reference_timestamp_ms,base_origin_x_m,base_origin_y_m,base_origin_z_m,base_reference_yaw_rad\n";
    }
    finished_ = false;
}

bool log_output::write(const ::camera &camera, const SolvedFrame &camera_poses,
                   const PredictionResult &prediction, const ControlTarget &control,
                   const SolvedFrame *base_poses, const ::pose *pose_base)
{
    for (const auto *poses : {&camera_poses, base_poses})
    {
        if (!poses)
            continue;
        const bool base_frame = poses == base_poses;
        auto &stream = base_frame ? base_ : raw_;
        for (int i = 0; i < static_cast<int>(poses->observations.size()); ++i)
        {
            const auto &armor = poses->armors.at(i);
            const auto &z = poses->observations[i];
            stream << camera.frame_id << ',' << camera.timestamp_ms;
            for (int j = 0; j < 3; ++j)
                stream << ',' << armor.tvec.at<double>(j);
            for (int j = 0; j < 4; ++j)
                stream << ',' << z(j);
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
    if (pose_base)
    {
        const auto &base_from_camera = pose_base->base_from_camera_;
        const Eigen::Quaterniond q(base_from_camera.linear());
        const auto &position = base_from_camera.translation();
        transforms_ << camera.frame_id << ',' << camera.timestamp_ms << ',' << pose_base->latest_state_.receive_timestamp_ms
                    << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z()
                    << ',' << position.x() << ',' << position.y() << ',' << position.z()
                    << ',' << pose_base->reference_time_ms_ << ",0,0,0," << pose_base->reference_yaw_rad_ << '\n';
    }
    controls_ << camera.frame_id << ',' << camera.timestamp_ms << ',' << control.yaw_rad << ',' << control.pitch_rad << ','
              << control.valid() << ',' << control.armor_id << ',' << control.status << '\n';
    return true;
}

void log_output::finish()
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
        for (int i = 0; i < static_cast<int>(names.size()); ++i)
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
    finished_ = true;
}

void log_output::write(std::int64_t frame_id, double timestamp_ms,
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
        for (int i = 0; i < static_cast<int>(prediction.future.plates.size()); ++i)
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

log_output::log_output(PredictorType model, const std::filesystem::path &directory,
               const std::string &suffix, double horizon_ms, bool base_frame)
{
    open_predictions(model, directory, suffix, horizon_ms, base_frame);
}

void log_output::open_predictions(PredictorType model, const std::filesystem::path &directory,
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
