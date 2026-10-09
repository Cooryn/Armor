#pragma once
#include "camera.hpp"
#include "gimbal.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

class PredictionOutput
{
public:
    PredictionOutput()
    {
        type = PREDICTOR_ARMOR;
        finished = true;
    }
    PredictionOutput(PredictorType model, const std::filesystem::path &directory,
                     const std::string &suffix, bool base_frame = false)
    {
        constexpr const char *basic_header =
            "frame_id,predicted_x,observed_x,error_x,predicted_z,observed_z,error_z,"
            "predicted_yaw,observed_yaw,error_yaw,predicted_distance,observed_distance,error_distance";
        constexpr const char *polar_header =
            "frame_id,xc,vxc,yc,vyc,zc,vzc,body_yaw,w,r,err_target_yaw,err_target_pitch,"
            "err_distance,err_armor_yaw,obs_armor_yaw";
        constexpr const char *armor_header =
            "frame_id,timestamp,prediction_horizon_ms,prediction_timestamp,future_xc,future_yc,future_zc,"
            "future_body_yaw,xc,yc,zc,vxc,vyc,vzc,w,xa,za,armor_id,body_yaw,pred_armor_yaw,obs_armor_yaw,"
            "err_target_yaw,err_target_pitch,err_distance,err_armor_yaw,r,dl,dh,observation_count,"
            "accepted_count,rejected_count,status";

        type = model;
        base_coordinates = base_frame;

        const std::array<const char *, 3> prefixes = {"", "polar_", "armor_"},
                                         headers = {basic_header, polar_header, armor_header};
        const char *prefix = prefixes[type], *header = headers[type];
        auto open = [](std::ofstream &csv, const std::filesystem::path &path, const char *header)
        {
            csv.open(path);
            if (!csv)
                throw std::runtime_error("Cannot write " + path.string());
            csv.exceptions(std::ios::badbit | std::ios::failbit);
            csv << header;
        };
        std::filesystem::create_directories(directory);
        open(results, directory / (std::string(prefix) + "prediction_result_" + suffix + ".csv"), header);
        metrics_path = directory / (std::string(prefix) + "rmse_result_" + suffix + ".txt");
        if (type == PREDICTOR_ARMOR)
        {
            open(futures, directory / ("armor_future_prediction_" + suffix + ".csv"),
                 "frame_id,timestamp,prediction_timestamp,prediction_horizon_ms,armor_id,x,y,z,"
                 "armor_orientation_yaw,source_status");
        }
        for (auto *csv : {&results, &futures})
            if (csv->is_open())
                *csv << (base_coordinates ? ",coordinate_frame\n" : "\n") << std::setprecision(17);
    }
    void write(std::int64_t frame, double timestamp,
               const VideoPrediction &prediction);
    void finish();

private:
    PredictorType type;
    std::filesystem::path metrics_path;
    std::ofstream results, futures;
    std::array<double, 4> squared_errors{};
    std::array<std::size_t, 4> error_counts{};
    bool finished = false;
    bool base_coordinates = false;
};


class Output
{
public:
    void open(const std::filesystem::path &root, const Camera &input, PredictorType model, bool preview);
    bool write(const Camera &input, const Solver &solver, const SolvedFrame &camera_poses,
               const SolvedFrame &base_poses, const PoseBase &pose_base,
               const VideoPrediction &prediction, const ControlTarget &command,
               std::chrono::steady_clock::time_point frame_start);
    void finish();
private:
    static constexpr const char *window = "Armor - current / future prediction - Esc to stop";
    PredictorType type_ = PREDICTOR_ARMOR;
    bool preview_ = true, finished_ = true;
    std::filesystem::path output_path_, raw_path_;
    cv::VideoWriter writer_;
    std::ofstream raw_, base_, transforms_, controls_;
    PredictionOutput predictions_;
    std::int64_t processed_ = 0;
};
