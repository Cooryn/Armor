#pragma once

#include <opencv2/core.hpp>
#include <vector>

// 【新增】：定义要打击的敌方颜色
enum class EnemyColor
{
    RED,
    BLUE
};

// 【修改】：将 thresholdRed 替换为通用的 extractColor
// 对原图做特定颜色阈值分割，返回二值掩码
cv::Mat extractColor(const cv::Mat &src, EnemyColor color);

// 从掩码中提取轮廓（最外层）
std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask);

// 将掩码与原图叠加，得到阈值处理可视化结果
cv::Mat applyMaskToImage(const cv::Mat &src, const cv::Mat &mask);

// 在原图上绘制所有轮廓
cv::Mat drawAllContours(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &contours);

// 按灯条形状（长宽比）过滤轮廓
std::vector<std::vector<cv::Point>> filterLightBars(const std::vector<std::vector<cv::Point>> &contours, double minAspectRatio = 3.0, double minArea = 50.0);

// 在原图上绘制灯条的最小外接旋转矩形
cv::Mat drawLightBarRects(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &lightBars);

// ==========================================
// 2. 为 PnP 保留的装甲板结构与匹配方法
// ==========================================

struct Armor
{
    cv::RotatedRect left_light;
    cv::RotatedRect right_light;
    cv::Point2f center;
    cv::Point2f vertices[4];
    cv::Mat rvec;
    cv::Mat tvec;
    double yaw = 0.0;
};

// 【新增桥接函数】：将你的 contours 转为 RotatedRect，并应用你的角度过滤逻辑
std::vector<cv::RotatedRect> getValidLightRects(const std::vector<std::vector<cv::Point>> &lightBars);

// 装甲板匹配与绘制
std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars);
cv::Mat drawArmors(const cv::Mat &src, const std::vector<Armor> &armors);