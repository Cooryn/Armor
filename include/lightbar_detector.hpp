#pragma once

#include <opencv2/core.hpp>
#include <vector>

// 对原图做红色阈值分割（BGR 单次判断，无需拼接），返回二值掩码
cv::Mat extractColor(const cv::Mat &src, EnemyColor color);

// 从掩码中提取轮廓（最外层）
std::vector<std::vector<cv::Point>> extractContours(const cv::Mat& mask);

// 将掩码与原图叠加，得到阈值处理可视化结果
cv::Mat applyMaskToImage(const cv::Mat& src, const cv::Mat& mask);

// 在原图上绘制所有轮廓
cv::Mat drawAllContours(const cv::Mat& src,
                        const std::vector<std::vector<cv::Point>>& contours);

// 按灯条形状（长宽比）过滤轮廓
std::vector<std::vector<cv::Point>> filterLightBars(
    const std::vector<std::vector<cv::Point>>& contours,
    double minAspectRatio = 3,
    double minArea = 50.0);

// 在原图上绘制灯条的最小外接旋转矩形
cv::Mat drawLightBarRects(const cv::Mat& src,
                          const std::vector<std::vector<cv::Point>>& lightBars);

enum class EnemyColor
{
    RED,
    BLUE
};

// 1. 定义装甲板数据结构
struct Armor
{
    cv::RotatedRect left_light;
    cv::RotatedRect right_light;
    cv::Point2f center;      // 装甲板中心点
    cv::Point2f vertices[4]; // 装甲板4个角点 (左上, 左下, 右下, 右上) - 为PnP做准备
};

// 2. 将轮廓转换为带有方向特征的灯条外接矩形
std::vector<cv::RotatedRect> getLightBarRects(const std::vector<std::vector<cv::Point>> &contours);

// 3. 核心匹配逻辑：输入灯条，输出装甲板
std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars);

// 4. 可视化绘制装甲板
cv::Mat drawArmors(const cv::Mat &src, const std::vector<Armor> &armors);
