#pragma once
#include "pose_base.hpp"

struct PredictionResult;

struct GimbalConfig
{
    double yaw_min_rad = -0.7853981633974483, yaw_max_rad = 0.7853981633974483;
    double pitch_min_rad = -0.5235987755982988, pitch_max_rad = 0.5235987755982988;
    double max_age_ms = 100;
};
struct ControlTarget
{
    float yaw_rad = 0, pitch_rad = 0;
    int armor_id = -1;
    std::string status = "no_target";
    bool valid() const { return status == "tracking"; }
};
class gimbal
{
public:
    static ControlTarget solve(const PredictionResult &prediction, const ::pose &pose_base,
                           double image_timestamp_ms, double now_ms, const GimbalConfig &gimbal_config = GimbalConfig{});
};
