#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <memory>

// ==========================================
// 1. 抽象基类 (统一接口)
// ==========================================
class InputStream
{
public:
    virtual ~InputStream() = default; // 虚析构，防止内存泄漏

    // 纯虚函数：获取一帧图像。返回 false 表示流结束或出错
    virtual bool getFrame(cv::Mat &frame) = 0;
};

// ==========================================
// 2. 视频文件输入流 (Video 1 & 2)
// ==========================================
class VideoFileStream : public InputStream
{
private:
    cv::VideoCapture cap;

public:
    explicit VideoFileStream(const std::string &path)
    {
        cap.open(path);
        if (!cap.isOpened())
        {
            std::cerr << "[Error] 无法打开视频文件: " << path << std::endl;
        }
    }

    bool getFrame(cv::Mat &frame) override
    {
        if (!cap.isOpened())
            return false;
        return cap.read(frame); // 视频播放完毕会自然返回 false
    }
};

// ==========================================
// 3. 物理摄像头输入流 (USB Camera)
// ==========================================
class CameraStream : public InputStream
{
private:
    cv::VideoCapture cap;

public:
    explicit CameraStream(int cam_id)
    {
        // 使用 CAP_DSHOW 在 Windows 下大幅加速摄像头启动并解决掉帧问题
        cap.open(cam_id, cv::CAP_DSHOW);

        if (cap.isOpened())
        {
            // RM 工业摄像头常见设置 (可根据后续实车硬件调整)
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cap.set(cv::CAP_PROP_FPS, 60);
            // cap.set(cv::CAP_PROP_EXPOSURE, -6); // RM 视觉中通常需要大幅降低曝光来凸显灯条
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

// ==========================================
// 4. 静态图像输入流 (Image 1 & 2)
// ==========================================
class ImageStream : public InputStream
{
private:
    cv::Mat static_image;

public:
    explicit ImageStream(const std::string &path)
    {
        static_image = cv::imread(path);
        if (static_image.empty())
        {
            std::cerr << "[Error] 无法读取图片: " << path << std::endl;
        }
    }

    bool getFrame(cv::Mat &frame) override
    {
        if (static_image.empty())
            return false;
        frame = static_image.clone();
        return true; // 恒定返回 true，方便主控程序写死 while 循环进行算法调试
    }
};

// ==========================================
// 5. ROS2 话题输入流 (预留接口)
// ==========================================
class ROS2Stream : public InputStream
{
private:
    cv::Mat current_frame;
    // std::mutex frame_mutex;
    // rclcpp::Node::SharedPtr node_;
    // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

public:
    ROS2Stream()
    {
        std::cout << "[Info] ROS2 流接口已就绪 (待挂载 cv_bridge 回调)" << std::endl;
    }

    // 预留的回调更新函数
    /*
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(frame_mutex);
        current_frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }
    */

    bool getFrame(cv::Mat &frame) override
    {
        // std::lock_guard<std::mutex> lock(frame_mutex);
        if (current_frame.empty())
            return false;
        frame = current_frame.clone();
        return true;
    }
};