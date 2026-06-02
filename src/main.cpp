#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <memory>
#include "input_stream.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"

int main()
{
    // ==========================================
    // 1. 运行参数配置
    // ==========================================
    std::string run_mode = "image"; // 【注意】这里设为了 "image" 方便测试出图
    std::string input_path = (run_mode == "video") ? "./assets/video/video_1.avi" : "./assets/image/image_1.jpg";
    std::string output_path = (run_mode == "video") ? "./results/video/video_1.mp4" : "./results/image/image_1.jpg";

    // 针对你的 3号平衡步兵，设置要打击的颜色 (看图应该是蓝方)
    EnemyColor target_color = EnemyColor::RED;

    // 自动创建输出文件夹
    std::filesystem::path out_dir = std::filesystem::path(output_path).parent_path();
    std::filesystem::create_directories(out_dir);

    // ==========================================
    // 2. 初始化输入输出流
    // ==========================================
    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
        stream = std::make_unique<VideoFileStream>(input_path);
    else if (run_mode == "image")
        stream = std::make_unique<ImageStream>(input_path);
    else
        return -1;

    cv::VideoWriter video_writer;
    cv::Mat frame;

    if (!stream->getFrame(frame))
    {
        std::cerr << "[错误] 无法读取视频/图片，请检查路径: " << input_path << std::endl;
        return -1;
    }

    if (run_mode == "video")
    {
        video_writer.open(output_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30.0, frame.size(), true);
    }

    // ==========================================
    // 3. PnP 解算器初始化
    // ==========================================
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307, 0, 645.344, 0, 1288.140, 483.616, 0, 0, 1);
    cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << -0.4756, 0.2183, 0.00049, -0.00034, 0);
    Solver pnp_solver(camera_matrix, distort_coeffs);

    int frame_count = 0;
    std::cout << "[信息] 视觉流水线启动..." << std::endl;

    // ==========================================
    // 4. 视觉处理主循环
    // ==========================================
    do
    {
        if (frame.empty())
            continue;

        // --- A. 图像预处理 ---
        cv::Mat processed_frame;
        cv::addWeighted(frame, 0.6, frame, 0.0, 0.0, processed_frame);

        // ==========================================
        // --- B. 2D 目标检测 (使用你的原生方法) ---
        // ==========================================
        // 1. 原生颜色提取
        cv::Mat mask = extractColor(processed_frame, target_color);

        // 2. 原生轮廓提取
        auto contours = extractContours(mask);

        // 3. 原生灯条过滤 (调用你的参数 minAspectRatio=3.0, minArea=50.0)
        auto lightBars = filterLightBars(contours, 1.5, 10.0);

        // 4. 将原生轮廓桥接为旋转矩形，用于后续 PnP
        auto lightRects = getValidLightRects(lightBars);

        // 5. 匹配装甲板
        auto armors = matchArmors(lightRects);
        // ==========================================

        // 如果你需要像之前一样保存轮廓和二值化中间图，可以直接调用：
        // cv::Mat debug_contours = drawAllContours(frame, contours);
        // cv::Mat debug_mask = applyMaskToImage(frame, mask);

        // --- C. 可视化渲染基础信息 ---
        cv::Mat result = drawArmors(frame, armors);

        // ... 接下来的 PnP 解算代码保持不变 ...

        // --- D. 3D 位姿解算 (PnP) ---
        for (auto &armor : armors)
        {
            if (pnp_solver.solve(armor))
            {
                pnp_solver.drawAxis(result, armor);
                cv::putText(result, cv::format("Z: %.2fm Yaw: %.1f deg", armor.tvec.at<double>(2), armor.yaw),
                            cv::Point(armor.vertices[0].x - 10, armor.vertices[0].y - 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }
        }

        // --- E. 输出呈现与保存 ---
        cv::imshow("RoboMaster Vision System - Final", result);

        if (run_mode == "video" && video_writer.isOpened())
        {
            video_writer.write(result);
        }
        else if (run_mode == "image")
        {
            std::filesystem::path out_file(output_path);
            std::string stem = out_file.stem().string();
            std::string ext = out_file.extension().string();
            std::string dir = out_file.parent_path().string();

            // 【核心修复】：先调用你的原生方法，把轮廓画到原图的克隆版上，生成一张图像 (cv::Mat)
            cv::Mat contours_img = drawAllContours(frame, contours);

            // 保存三张图：传入的必须都是 cv::Mat
            cv::imwrite(dir + "/" + stem + "_mask" + ext, mask);
            cv::imwrite(dir + "/" + stem + "_contours" + ext, contours_img); // 传画好的图！
            cv::imwrite(output_path, result);

            std::cout << "[信息] 已保存过程图: " << stem << "_mask" << ext << " 和 " << stem << "_contours" << ext << std::endl;
        }

        frame_count++;

        if (cv::waitKey((run_mode == "image") ? 0 : 30) == 27)
            break;

    } while (stream->getFrame(frame));

    if (video_writer.isOpened())
        video_writer.release();
    cv::destroyAllWindows();
    std::cout << "[信息] 处理完成！" << std::endl;

    return 0;
}