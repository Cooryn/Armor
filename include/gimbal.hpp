#pragma once
#include "pose_base.hpp"

struct GimbalConfig
{
    double yaw_min_rad = -0.7853981633974483, yaw_max_rad = 0.7853981633974483;
    double pitch_min_rad = -0.5235987755982988, pitch_max_rad = 0.5235987755982988;
    double max_age_ms = 100;
    double max_position_variance = 1;
};
struct ControlTarget
{
    float yaw_rad = 0, pitch_rad = 0;
    bool valid = false;
    int armor_id = -1;
    std::string status = "no_target";
};
ControlTarget solve_gimbal(const VideoPrediction &prediction, const PoseBase &pose_base,
                           double image_timestamp_ms, double now_ms, const GimbalConfig &gimbal_config = GimbalConfig{});
