#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>
#include <string>

enum CameraSource { CAMERA_VIDEO, CAMERA_OPENCV };
struct CameraConfig
{
    CameraSource source = CAMERA_VIDEO;
    std::filesystem::path video;
    int device = 0;
    double recording_fps = 30; // Explicit recording rate for the live camera.
};
class Camera
{
public:
    void open(const CameraConfig &config);
    bool next();
    bool live() const;

    cv::Mat image_;
    std::filesystem::path path_;
    std::string stem_, suffix_;
    double fps_ = 0, timestamp_ms = 0;
    std::int64_t frame_id_ = -1;
private:
    CameraSource source_ = CAMERA_VIDEO;
    cv::VideoCapture stream_;
    double frame_count_ = 0;
};
