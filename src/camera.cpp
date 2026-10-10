#include "camera.hpp"
#include "serial.hpp"
#include <stdexcept>

constexpr int camera_width = 1280;
constexpr int camera_height = 720;
constexpr double camera_capture_fps = 60;

bool camera::live() const
{
    return source_ == CAMERA;
}

void camera::open(CameraSource source, const std::filesystem::path &video_path, int camera_device, double recording_fps)
{
    stream_.release();
    source_ = source;
    path_ = video_path;
    frame_id_ = -1;
    frame_count_ = 0;
    if (source_ == VIDEO)
    {
        stream_.open(path_.string());
        stem_ = path_.stem().string();
        frame_count_ = stream_.get(cv::CAP_PROP_FRAME_COUNT);
        fps_ = stream_.get(cv::CAP_PROP_FPS);
    }
    else
    {
#ifdef _WIN32
        stream_.open(camera_device, cv::CAP_DSHOW);
#else
        stream_.open(camera_device);
#endif
        stem_ = "camera_live";
        fps_ = recording_fps;
    }
    if (!stream_.isOpened())
        throw std::runtime_error("Cannot open image source: " + stem_);
    if (live())
    {
        if (!stream_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')) ||
            !stream_.set(cv::CAP_PROP_FRAME_WIDTH, camera_width) ||
            !stream_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height) ||
            !stream_.set(cv::CAP_PROP_FPS, camera_capture_fps))
            throw std::runtime_error("Cannot configure USB camera: MJPG 1280x720 at 60 FPS");
    }
    const auto separator = stem_.find_last_of('_');
    suffix_ = separator == std::string::npos ? stem_ : stem_.substr(separator + 1);
    if (!next())
        throw std::runtime_error("Cannot read the first image: " + stem_);
}

bool camera::next()
{
    if (!stream_.read(image_))
    {
        if (live() || frame_id_ < 0 || double(frame_id_ + 1) < frame_count_)
            throw std::runtime_error("Image acquisition failed before the end of the source");
        return false;
    }
    ++frame_id_;
    timestamp_ms = live() ? serial::monotonic_time_ms() : double(frame_id_) * 1000 / fps_;
    return true;
}
