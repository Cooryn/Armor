#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "predictor.hpp"
#include "camera_tracking.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Modify these values and rebuild Armor. Paths are relative to the project root.
constexpr EnemyColor target_color = ENEMY_RED;
constexpr const char *video_file = "assets/video/video_2.avi";
constexpr PredictorType predictor_type = PREDICTOR_ARMOR;
constexpr bool preview = true;
constexpr double prediction_horizon_ms = 50;

int main(int argc, char **)
{
    try
    {
        if (argc != 1)
            throw std::invalid_argument("Armor takes no arguments; edit the configuration in src/Armor.cpp and rebuild");
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        VideoInput video(root / std::filesystem::u8path(video_file));
        Solver pnp;
        VideoPredictor predictor{predictor_type, prediction_horizon_ms};
        VideoOutput output(root, video, predictor_type, preview);
        do
        {
            const auto frame_start = std::chrono::steady_clock::now();
            auto detections = detectArmors(video.image_, target_color);
            const auto poses = pnp.solve_frame(std::move(detections), video.timestamp_ms, video.stem_);
            const auto prediction = predictor.update(video.frame_id_, video.timestamp_ms, poses.observations);
            if (!output.write(video, pnp, poses, prediction, frame_start))
                break;
        } while (video.next());
        output.finish();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Armor failed: " << error.what() << '\n';
        return 1;
    }
}
