#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

class InputStream
{
public:
    virtual ~InputStream() = default;
    virtual bool getFrame(cv::Mat &frame) = 0;
};

class VideoFileStream : public InputStream
{
private:
    cv::VideoCapture cap;

public:
    explicit VideoFileStream(const std::string &path)
    {
        cap.open(path);
        if (!cap.isOpened())
            std::cerr << "[Error] 无法打开视频文件: " << path << std::endl;
    }
    bool getFrame(cv::Mat &frame) override
    {
        if (!cap.isOpened())
            return false;
        return cap.read(frame);
    }
};

class CameraStream : public InputStream
{
private:
    cv::VideoCapture cap;

public:
    explicit CameraStream(int cam_id)
    {
        cap.open(cam_id, cv::CAP_DSHOW);
        if (cap.isOpened())
        {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cap.set(cv::CAP_PROP_FPS, 60);
        }
        else
        {
            std::cerr << "[Error] 无法打开摄像头设备: " << cam_id << std::endl;
        }
    }
    bool getFrame(cv::Mat &frame) override
    {
        if (!cap.isOpened())
            return false;
        return cap.read(frame);
    }
};

class ImageStream : public InputStream
{
private:
    cv::Mat static_image;

public:
    explicit ImageStream(const std::string &path)
    {
        static_image = cv::imread(path);
        if (static_image.empty())
            std::cerr << "[Error] 无法读取图片: " << path << std::endl;
    }
    bool getFrame(cv::Mat &frame) override
    {
        if (static_image.empty())
            return false;
        frame = static_image.clone();
        return true;
    }
};