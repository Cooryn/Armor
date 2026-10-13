#include "lightbar_detector.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

struct Parameters {
    int color, gray;
    double area, shape;
    float length, offset, angle, score;
    bool reject_clipped = false;
    bool high_threshold_first = false;
};
// Baseline, segmentation-only variants, then combined segmentation/pairing candidates.
static const std::vector<Parameters> candidates{
    {70,170,40,1.5,2,.8f,20,.35f}, {40,120,40,1.5,2,.8f,20,.35f},
    {40,120,12,1.5,2,.8f,20,.35f}, {40,120,12,2,1.6f,.4f,20,.4f},
    {40,120,12,2,1.8f,.5f,20,.4f}, {35,100,12,2,1.8f,.5f,20,.4f},
    {50,130,12,2,1.8f,.5f,20,.4f}, {40,120,20,2,1.8f,.5f,20,.4f},
    {40,120,12,1.8,1.8f,.5f,15,.4f}, {40,120,12,2,1.8f,.4f,15,.45f},
    {40,120,12,1.5,1.8f,.5f,20,.4f}, {40,120,12,2,1.8f,.5f,20,.35f},
    {40,100,12,1.5,2,.5f,20,.45f}, {40,100,12,1.5,2,.5f,20,.4f},
    {40,100,12,1.8,2,.5f,20,.4f}, {50,100,12,1.5,2,.5f,20,.4f},
    {35,80,12,1.5,2,.5f,20,.4f}, {40,100,12,1.5,2,.4f,20,.45f},
    {70,100,12,1.5,2,.5f,20,.4f}, {60,100,12,1.5,2,.5f,20,.4f},
    {55,120,12,1.5,2,.5f,20,.4f}, {55,100,12,1.5,2,.5f,20,.4f},
    {70,120,12,1.5,2,.5f,20,.4f},
    {40,100,12,1.5,2,.5f,20,.4f,true},
    {70,170,12,1.5,2,.5f,20,.4f,true},
    {40,100,12,1.5,2,.5f,20,.4f,true,true}
};

int main(int argc, char **argv) {
    try {
        if (argc < 3) throw std::runtime_error("Expected repository root, output directory, optional video stride");
        const std::filesystem::path root(argv[1]), directory(argv[2]);
        const int stride = argc > 3 ? std::stoi(argv[3]) : 5;
        const bool finalists_only = argc > 4;
        std::filesystem::create_directories(directory);
        std::ofstream out(directory / "detections.csv");
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << "source,frame,variant,candidate,lights,count,index,cx,cy,x0,y0,x1,y1,x2,y2,x3,y3\n" << std::setprecision(9);
        const auto process = [&](const cv::Mat &image, const std::string &source, int frame, EnemyColor color, const std::string &variant) {
            for (int id = 0; id < static_cast<int>(candidates.size()); ++id) {
                if (finalists_only && id != 0 && id < 23) continue;
                const auto &p = candidates[id];
                std::vector<::armor> armors;
                int light_count = 0;
                for (int pass = 0; pass < (p.high_threshold_first ? 2 : 1); ++pass) {
                    auto mask = detector::extract_color(image, color,
                        p.high_threshold_first && pass == 0 ? 70 : p.color,
                        p.high_threshold_first && pass == 0 ? 170 : p.gray);
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                    if (p.reject_clipped)
                        contours.erase(std::remove_if(contours.begin(), contours.end(), [&](const auto &contour) {
                            const auto box = cv::boundingRect(contour);
                            return box.x <= 0 || box.y <= 0 || box.br().x >= image.cols || box.br().y >= image.rows;
                        }), contours.end());
                    std::vector<float> quality;
                    auto lights = detector::get_valid_light_rects(contours, quality, 55, p.shape, p.area);
                    armors = detector::match_armors(lights, quality, p.angle, p.length, .8f, p.offset, 3.1f, p.score);
                    light_count = static_cast<int>(lights.size());
                    if (!armors.empty()) break;
                }
                if (id == 25) {
                    const auto actual = detector::detect(image, color);
                    if (actual.size() != armors.size()) throw std::runtime_error("Production detector count differs from candidate 25");
                    for (int j = 0; j < static_cast<int>(actual.size()); ++j)
                        for (int k = 0; k < 4; ++k)
                            if (cv::norm(actual[j].vertices[k] - armors[j].vertices[k]) > 1e-5)
                                throw std::runtime_error("Production detector corners differ from candidate 25");
                }
                for (int j = 0; j < std::max(1, static_cast<int>(armors.size())); ++j) {
                    out << source << ',' << frame << ',' << variant << ',' << id << ',' << light_count << ',' << armors.size();
                    if (armors.empty()) out << ",-1,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan\n";
                    else {
                        const auto &a = armors[j];
                        out << ',' << j << ',' << a.center.x/image.cols << ',' << a.center.y/image.rows;
                        for (const auto &v : a.vertices) out << ',' << v.x/image.cols << ',' << v.y/image.rows;
                        out << '\n';
                    }
                }
            }
        };
        for (int i = 0; i <= 10; ++i) {
            const std::string name = "image_" + std::to_string(i);
            const auto image = cv::imread((root / "assets/image" / (name + ".jpg")).string());
            if (image.empty()) throw std::runtime_error("Missing image " + name);
            const auto color = i == 1 || i == 2 ? ENEMY_BLUE : ENEMY_RED;
            process(image, name, 0, color, "original");
            cv::Mat dim, small;
            image.convertTo(dim, -1, .6);
            cv::resize(image, small, {}, .5, .5, cv::INTER_AREA);
            process(dim, name, 0, color, "dim");
            process(small, name, 0, color, "small");
            process(image, name, 0, color == ENEMY_RED ? ENEMY_BLUE : ENEMY_RED, "wrong_color");
            // Manually reviewed upper image region contains no armor plates.
            process(image(cv::Rect(0, 0, image.cols, int(image.rows * .45))), name, 0, color, "background");
        }
        for (int i : {1,2}) {
            const std::string name = "video_" + std::to_string(i);
            cv::VideoCapture cap((root / "assets/video" / (name + ".avi")).string());
            if (!cap.isOpened()) throw std::runtime_error("Missing video " + name);
            cv::Mat frame;
            int index = 0;
            while (cap.read(frame)) {
                if (index % stride == 0) process(frame, name, index, ENEMY_RED, "original");
                ++index;
            }
            std::cout << name << ": " << index << " frames, stride " << stride << '\n';
        }
        out.close();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
