#pragma once
#include "log_output.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

class video_output
{
public:
    video_output(const std::filesystem::path &directory, const ::camera &camera)
        : source_(camera.source), fps_(camera.fps)
    {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        cv::resizeWindow(window, 1280, 720);
        path_ = directory / (camera.stem + ".avi");
        std::filesystem::create_directories(directory);
        if (source_ == VIDEO && std::filesystem::weakly_canonical(camera.path) == std::filesystem::weakly_canonical(path_))
            throw std::invalid_argument("Output video must not overwrite the input video");
        writer_.open(path_.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps_, camera.image.size());
        if (!writer_.isOpened())
            throw std::runtime_error("Cannot write video: " + path_.string());
    }

    void record(const ::camera &camera, const cv::Mat &frame)
    {
        if (source_ == VIDEO)
        {
            writer_.write(frame);
            ++frames_;
            return;
        }
        if (previous_canvas_.empty())
            first_timestamp_ms_ = camera.timestamp_ms;
        else
        {
            const auto target_frame = static_cast<std::int64_t>(std::ceil((camera.timestamp_ms - first_timestamp_ms_) * fps_ / 1000 - 1e-9));
            while (frames_ < target_frame)
            {
                writer_.write(previous_canvas_);
                ++frames_;
            }
        }
        previous_canvas_ = frame.clone();
    }

    void finish()
    {
        if (!writer_.isOpened())
            return;
        if (!previous_canvas_.empty())
        {
            writer_.write(previous_canvas_);
            ++frames_;
        }
        writer_.release();
        cv::destroyWindow(window);
        cv::VideoCapture exported(path_.string());
        if (!exported.isOpened() || exported.get(cv::CAP_PROP_FRAME_COUNT) != double(frames_))
            throw std::runtime_error("Output video write failed or frame count is incomplete: " + path_.string());
    }

    bool write(const ::camera &camera, const ::solver &solver, const SolvedFrame &camera_poses,
               std::chrono::steady_clock::time_point frame_start,
               const PredictionResult *prediction = nullptr, PredictorType model = PREDICTOR_SINGLE_PLATE,
               const ControlTarget *control = nullptr, const ::pose *pose_base = nullptr)
    {
        canvas = camera.image.clone();
        detector::draw_armors(canvas, camera_poses.armors);
        const char *label = prediction ? (model == PREDICTOR_SINGLE_PLATE ? "SinglePlate" : "Armor") : "Detect";
        std::string info = cv::format("%s %s frame %lld", camera.source == CAMERA ? "LIVE" : "REPLAY", label,
                                      static_cast<long long>(camera.frame_id));
        if (prediction)
        {
            Eigen::Isometry3d camera_from_base = Eigen::Isometry3d::Identity();
            if (pose_base)
                camera_from_base = pose_base->base_from_camera_.inverse();
            draw_prediction(canvas, prediction->current, camera_from_base, solver, false);
            draw_prediction(canvas, prediction->future, camera_from_base, solver, true);
            info += " " + prediction->status;
            if (pose_base && control)
                info += "  control " + control->status;
        }
        detector::draw_video_info(canvas, camera_poses.armors, info);
        record(camera, canvas);
        return show_canvas(camera, frame_start);
    }

private:
    inline static const char *const window = "Armor video - Esc to stop";
    cv::Mat canvas;
    bool show_canvas(const ::camera &camera, std::chrono::steady_clock::time_point frame_start);
    static void draw_prediction(cv::Mat &image, const PredictionGeometry &geometry,
                                const Eigen::Isometry3d &camera_from_base, const ::solver &solver, bool future);
    CameraSource source_;
    double fps_, first_timestamp_ms_ = 0;
    std::int64_t frames_ = 0;
    std::filesystem::path path_;
    cv::VideoWriter writer_;
    cv::Mat previous_canvas_;
};


inline void video_output::draw_prediction(cv::Mat &image, const PredictionGeometry &geometry, const Eigen::Isometry3d &camera_from_base,
                             const ::solver &solver, bool future)
{
    const std::array<cv::Scalar, 4> colors = {cv::Scalar(0, 255, 255), cv::Scalar(255, 180, 0),
                                              cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 100)};
    const auto project = [&](const Eigen::Vector3d &base_point) -> std::optional<cv::Point>
    {
        const Eigen::Vector3d position = camera_from_base * base_point;
        if (position.z() <= .1)
            return std::nullopt;
        std::vector<cv::Point2d> pixels;
        cv::projectPoints(std::vector<cv::Point3d>{{position.x(), position.y(), position.z()}}, cv::Vec3d(0, 0, 0),
                          cv::Vec3d(0, 0, 0), solver.camera_matrix, solver.distort_coeffs, pixels);
        const auto &pixel = pixels.front();
        return cv::Point(cvRound(pixel.x), cvRound(pixel.y));
    };
    const auto marker = [&](const Eigen::Vector3d &point, const cv::Scalar &color)
    {
        const auto pixel = project(point);
        if (!pixel || !cv::Rect(0, 0, image.cols, image.rows).contains(*pixel))
            return;
        if (future)
            cv::drawMarker(image, *pixel, color, cv::MARKER_DIAMOND, 14, 2, cv::LINE_AA);
        else
            cv::circle(image, *pixel, 5, color, 2, cv::LINE_AA);
    };
    if (geometry.center)
        marker(*geometry.center, {255, 255, 255});
    for (int i = 0; i < static_cast<int>(geometry.plates.size()); ++i)
    {
        const auto &plate = geometry.plates[i];
        const Eigen::Vector3d center = plate.head<3>();
        const auto &color = colors[i % colors.size()];
        marker(center, color);
        const Eigen::Vector3d half_width(.0675 * std::cos(plate(3)), 0, .0675 * std::sin(plate(3))), half_height(0, .028, 0);
        const std::array<Eigen::Vector3d, 4> corners = {center - half_width - half_height, center + half_width - half_height,
                                                        center + half_width + half_height, center - half_width + half_height};
        std::array<std::optional<cv::Point>, 4> pixels;
        for (int j = 0; j < 4; ++j)
            pixels[j] = project(corners[j]);
        for (int j = 0; j < 4; ++j)
            if (pixels[j] && pixels[(j + 1) % 4])
            {
                auto a = *pixels[j], b = *pixels[(j + 1) % 4];
                if (!cv::clipLine(image.size(), a, b))
                    continue;
                if (!future)
                    cv::line(image, a, b, color, 2, cv::LINE_AA);
                else
                {
                    const double length = cv::norm(b - a);
                    if (length < 1)
                        continue;
                    const cv::Point2d delta = cv::Point2d(b - a) / length;
                    for (double start = 0; start < length; start += 12)
                        cv::line(image, cv::Point(cv::Point2d(a) + start * delta),
                                 cv::Point(cv::Point2d(a) + std::min(start + 7, length) * delta), color, 2, cv::LINE_AA);
                }
            }
    }
}

inline bool video_output::show_canvas(const ::camera &camera,
                          std::chrono::steady_clock::time_point frame_start)
{
    cv::imshow(window, canvas);
    const auto deadline = frame_start + std::chrono::duration<double>(camera.source == CAMERA ? 0 : 1 / camera.fps);
    do
    {
        const double remaining = std::chrono::duration<double, std::milli>(deadline - std::chrono::steady_clock::now()).count();
        if ((cv::waitKey(static_cast<int>(std::clamp(std::ceil(remaining), 1.0, 10.0))) & 0xff) == 27)
            return false;
        try
        {
            if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1)
                return false;
        }
        catch (const cv::Exception &error)
        {
            if (error.code == cv::Error::StsNullPtr)
                return false;
            throw;
        }
    } while (std::chrono::steady_clock::now() < deadline);
    return true;
}


