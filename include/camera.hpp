#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>
#include <string>

enum CameraSource
{
    VIDEO,
    CAMERA
};
class camera
{
public:
    void open(CameraSource input_source, const std::filesystem::path &video_path, int camera_device = 0, double recording_fps = 30);
    bool next();

    CameraSource source = VIDEO;
    cv::Mat image;
    std::filesystem::path path;
    std::string stem, suffix;
    double fps = 0, timestamp_ms = 0, read_duration_ms = 0;
    std::int64_t frame_id = -1;

private:
    cv::VideoCapture stream;
    double frame_count = 0;
};
