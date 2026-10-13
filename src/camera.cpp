#include "camera.hpp"
#include "serial.hpp"
#include <stdexcept>
#include <chrono>
#include <cmath>

const int camera_width = 1280;
const int camera_height = 720;
const double camera_capture_fps = 30;

void camera::open(CameraSource input_source, const std::filesystem::path &video_path, int camera_device, double recording_fps)
{
    stream.release();
    frame_id = -1;
    frame_count = 0;
    read_duration_ms = 0;
    source = input_source;
    this->path = video_path;
    if (source == VIDEO)
    {
        stream.open(path.string());
        stem = path.stem().string();
        frame_count = stream.get(cv::CAP_PROP_FRAME_COUNT);
        fps = stream.get(cv::CAP_PROP_FPS);
    }
    else
    {
        stream.open(camera_device, cv::CAP_DSHOW);
        stem = "camera_live";
        fps = recording_fps;
    }
    if (!stream.isOpened())
        throw std::runtime_error("Cannot open image source: " + stem);
    if (source == CAMERA)
    {
        const int mjpg = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        stream.set(cv::CAP_PROP_FRAME_WIDTH, camera_width);
        stream.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height);
        stream.set(cv::CAP_PROP_FPS, camera_capture_fps);
        stream.set(cv::CAP_PROP_FOURCC, mjpg);
        const int width = static_cast<int>(stream.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(stream.get(cv::CAP_PROP_FRAME_HEIGHT));
        const int format = static_cast<int>(stream.get(cv::CAP_PROP_FOURCC));
        if (width != camera_width || height != camera_height || format != mjpg)
            throw std::runtime_error(cv::format("Unexpected camera format: %dx%d FOURCC=%d; expected 1280x720 MJPG",
                                                width, height, format));
    }
    if (!std::isfinite(fps) || fps <= 0)
        throw std::runtime_error("Video/recording FPS must be finite and positive");
    const auto separator = stem.find_last_of('_');
    suffix = separator == std::string::npos ? stem : stem.substr(separator + 1);
    next();
}

bool camera::next()
{
    const auto read_start = std::chrono::steady_clock::now();
    if (!stream.read(image))
    {
        if (source == CAMERA || frame_id < 0 || double(frame_id + 1) < frame_count)
            throw std::runtime_error("Image acquisition failed before the end of the source");
        return false;
    }
    read_duration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - read_start).count();
    ++frame_id;
    timestamp_ms = source == CAMERA ? serial::monotonic_time_ms() : double(frame_id) * 1000 / fps;
    return true;
}
