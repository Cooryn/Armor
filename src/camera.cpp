#include "camera.hpp"
#include "serial.hpp"
#include <stdexcept>

bool Camera::live() const
{
    return source_ == CAMERA;
}

void Camera::open(const CameraConfig &config)
{
    stream_.release();
    source_ = config.source;
    path_ = config.video;
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
        stream_.open(config.device);
        stem_ = "camera_live";
        fps_ = config.recording_fps;
    }
    if (!stream_.isOpened())
        throw std::runtime_error("Cannot open image source: " + stem_);
    const auto separator = stem_.find_last_of('_');
    suffix_ = separator == std::string::npos ? stem_ : stem_.substr(separator + 1);
    if (!next())
        throw std::runtime_error("Cannot read the first image: " + stem_);
}

bool Camera::next()
{
    if (!stream_.read(image_))
    {
        if (live() || frame_id_ < 0 || double(frame_id_ + 1) < frame_count_)
            throw std::runtime_error("Image acquisition failed before the end of the source");
        return false;
    }
    ++frame_id_;
    timestamp_ms = live() ? monotonic_time_ms() : double(frame_id_) * 1000 / fps_;
    return true;
}
