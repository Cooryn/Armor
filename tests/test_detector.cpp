#include "lightbar_detector.hpp"
#include "solver.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

cv::Point2f left_center(const ::armor &armor) {
    return (armor.vertices[0] + armor.vertices[1]) / 2.f;
}

cv::Point2f right_center(const ::armor &armor) {
    return (armor.vertices[2] + armor.vertices[3]) / 2.f;
}

int main(int argc, char **argv) {
    try {
        // Dim, small and border-touching plates must survive; displaced light pairs must not.
        for (const auto color : {ENEMY_RED, ENEMY_BLUE}) {
            const auto paint = [color](int value) {
                return color == ENEMY_RED ? cv::Scalar(10, 20, value) : cv::Scalar(value, 20, 10);
            };
            cv::Mat dim = cv::Mat::zeros(120, 160, CV_8UC3);
            cv::rectangle(dim, {40,40}, {44,60}, paint(150), cv::FILLED);
            cv::rectangle(dim, {88,40}, {92,60}, paint(150), cv::FILLED);
            require(detector::detect(dim, color).size() == 1, "Dim plate missing");
            require(detector::detect(dim, color == ENEMY_RED ? ENEMY_BLUE : ENEMY_RED).empty(), "Wrong-color plate accepted");
            cv::Mat small = cv::Mat::zeros(120, 160, CV_8UC3);
            cv::rectangle(small, {40,40}, {42,48}, paint(240), cv::FILLED);
            cv::rectangle(small, {60,40}, {62,48}, paint(240), cv::FILLED);
            require(detector::detect(small, color).size() == 1, "Small plate missing");
            cv::Mat displaced = cv::Mat::zeros(120, 160, CV_8UC3);
            cv::rectangle(displaced, {40,40}, {44,60}, paint(240), cv::FILLED);
            cv::rectangle(displaced, {88,52}, {92,72}, paint(240), cv::FILLED);
            require(detector::detect(displaced, color).empty(), "Displaced clutter paired");
            cv::Mat border_plate = cv::Mat::zeros(120, 160, CV_8UC3);
            cv::rectangle(border_plate, {40,0}, {44,20}, paint(240), cv::FILLED);
            cv::rectangle(border_plate, {88,0}, {92,20}, paint(240), cv::FILLED);
            require(detector::detect(border_plate, color).size() == 1, "Border-touching plate missing");
        }
        std::vector<cv::RotatedRect> bars{
            {{513.8f,725.8f},{37.1f,6.3f},104.f},
            {{637.f,716.f},{32.f,10.f},90.f},
            {{713.7f,715.3f},{31.5f,15.3f},74.7f},
            {{821.4f,722.5f},{29.8f,5.3f},76.f}};
        auto selected = detector::match_armors(bars, std::vector<float>(bars.size(), 1.f), 20, 2, .8f, .8f);
        require(selected.size() == 1, "Expected one consistent pair");
        require(std::abs(left_center(selected[0]).x - 637.f) < 1,
                "Wrong left light for frame-753 geometry");
        require(std::abs(right_center(selected[0]).x - 713.7f) < 1,
                "Wrong right light for frame-753 geometry");
        auto expected = selected[0].center;
        std::reverse(bars.begin(), bars.end());
        selected = detector::match_armors(bars, std::vector<float>(bars.size(), 1.f), 20, 2, .8f, .8f);
        require(cv::norm(selected[0].center - expected) < 1e-5, "Input-order dependence");

        bars.clear();
        for (float x : {0.f, 72.f, 150.f, 222.f})
            bars.emplace_back(cv::Point2f(x,100), cv::Size2f(30,4), 90.f);
        selected = detector::match_armors(bars, std::vector<float>(bars.size(), 1.f));
        require(selected.size() == 2, "Expected two non-conflicting plates");
        require(right_center(selected[0]).x == 72.f &&
                left_center(selected[1]).x == 150.f, "Incorrect global assignment");
        std::vector<cv::RotatedRect> competing;
        for (float x : {0.f, 60.f, 120.f, 180.f})
            competing.emplace_back(cv::Point2f(x,100), cv::Size2f(30,4), 90.f);
        auto greedy = detector::match_armors(competing,{.6f,1.f,1.f,.6f},20,2,.8f,.8f,3.1f,.35f);
        require(greedy.size() == 1 && left_center(greedy[0]).x == 60.f &&
                right_center(greedy[0]).x == 120.f, "Greedy must prefer the highest-score pair");
        std::reverse(competing.begin(), competing.end());
        auto reversed = detector::match_armors(competing,{.6f,1.f,1.f,.6f},20,2,.8f,.8f,3.1f,.35f);
        require(reversed.size() == 1 && left_center(reversed[0]).x == 60.f &&
                right_center(reversed[0]).x == 120.f, "Greedy input-order dependence");
        require(detector::match_armors({}, {}).empty(), "Empty input");

        bars = {{{0,120},{30,4},90}, {{72,100},{30,4},90}, {{144,100},{30,4},90}};
        selected = detector::match_armors(bars, std::vector<float>(bars.size(), 1.f));
        require(selected.size() == 1 && left_center(selected[0]).x == 72,
                "First-fit won over the higher quality pair");

        bars = {{{0,100},{30,4},90}, {{36,200},{30,4},90}, {{72,100},{30,4},90}};
        require(detector::match_armors(bars, std::vector<float>(bars.size(), 1.f)).empty(),
                "Non-adjacent light bars must not form a plate");

        std::vector<std::vector<cv::Point>> outlines{{{97,80},{103,80},
            {103,98},{108,100},{103,102},{103,120},{97,120}}};
        cv::Mat spur_mask = cv::Mat::zeros(200, 200, CV_8UC1);
        cv::fillPoly(spur_mask, outlines, cv::Scalar(255));
        cv::findContours(spur_mask, outlines, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        std::vector<float> quality;
        auto fitted = detector::get_valid_light_rects(outlines, quality, 55);
        require(fitted.size() == 1 && quality.size() == 1, "Missing fitted light");
        require(std::abs(std::max(fitted[0].size.width, fitted[0].size.height)-40) < 1,
                "Light length was shortened, biasing PnP depth");
        require(quality[0] > 0 && quality[0] < 1, "Irregular contour needs a quality penalty");
        fitted = detector::get_valid_light_rects({{}, {{1,1}}}, quality, 55);
        require(fitted.empty() && quality.empty(), "Degenerate contours not skipped");

        bars = {{{0,100},{30,4},90}, {{72,100},{30,4},90}, {{144,100},{30,4},90}};
        selected = detector::match_armors(bars,{.3f,1.f,1.f},20,1.5f,1.2f,.8f,3.1f,.35f);
        require(selected.size() == 1 && left_center(selected[0]).x == 72,
                "Light quality did not affect ambiguous pairing");
        std::reverse(bars.begin(), bars.end());
        selected = detector::match_armors(bars,{1.f,1.f,.3f},20,1.5f,1.2f,.8f,3.1f,.35f);
        require(selected.size() == 1 && left_center(selected[0]).x == 72,
                "Light quality detached from sorted geometry");

        cv::Mat K = (cv::Mat_<double>(3,3) << 1000,0,640,0,1000,480,0,0,1);
        cv::Mat distortion = cv::Mat::zeros(1,5,CV_64F);
        std::vector<cv::Point3f> object{{-.0675f,-.028f,0}, {-.0675f,.028f,0},
                                      {.0675f,.028f,0}, {.0675f,-.028f,0}};
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object, cv::Vec3d(0,.4,0), cv::Vec3d(.1,.2,3), K, distortion, projected);
        ::armor armor;
        std::copy(projected.begin(), projected.end(), armor.vertices);
        ::solver solver{K, distortion};
        require(solver.solve(armor), "Rejected exact physical rectangle");
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(object, armor.rvec, armor.tvec, K, distortion, reprojected);
        require(cv::norm(projected, reprojected, cv::NORM_L2) / std::sqrt(projected.size()) < .01,
                "Unexpected reprojection error");
        require(std::abs(armor.yaw+ .4*180/CV_PI) < .01, "PnP yaw sign changed");
        cv::Mat lens = (cv::Mat_<double>(1,5) << -.15,.03,.001,-.002,0);
        ::solver distorted_solver{K, lens};
        for (double yaw : {-.9, -.3, .001, .3, .9}) {
            cv::Vec3d rotation(.08, yaw, -.03), position(.15,.1,2.5);
            cv::projectPoints(object, rotation, position, K, lens, projected);
            ::armor sample;
            std::copy(projected.begin(), projected.end(), sample.vertices);
            require(distorted_solver.solve(sample), "Exact distorted plate rejected");
            cv::Mat expected_rotation, recovered_rotation;
            cv::Rodrigues(rotation, expected_rotation);
            cv::Rodrigues(sample.rvec, recovered_rotation);
            require(cv::norm(expected_rotation-recovered_rotation) < .002,
                    "Wrong planar pose branch for exact synthetic plate");
            require(cv::norm(sample.tvec-cv::Mat(position)) < .001, "PnP translation changed");
        }

        ::armor first = armor, second = armor;
        first.center = {100,100}; second.center = {300,100};
        first.yaw = -20; second.yaw = 30;
        for (auto *a : {&first, &second}) {
            a->vertices[0] = a->center+cv::Point2f(-40,-15);
            a->vertices[1] = a->center+cv::Point2f(-40,15);
            a->vertices[2] = a->center+cv::Point2f(40,15);
            a->vertices[3] = a->center+cv::Point2f(40,-15);
        }
        solver.finish_frame({first,second}, 0);
        auto hints = solver.yaw_hints({second,first}, 1./30);
        require(hints[0] == 30 && hints[1] == -20, "Temporal matching depends on order");
        hints = solver.yaw_hints({first,first}, 1./30);
        require(hints[0] == -20 && hints[1] == -20, "Nearest matching failed to reuse prior");
        hints = solver.yaw_hints({first}, .2);
        require(!std::isfinite(hints[0]), "Stale pose used after long time gap");
        solver.finish_frame({}, 1./30);
        hints = solver.yaw_hints({first}, 2./30);
        require(!std::isfinite(hints[0]), "Missing frame failed to clear pose history");

        if (argc > 1) {
            const auto clear_blue = cv::imread(std::string(argv[1]) + "/assets/image/image_2.jpg");
            require(!clear_blue.empty() && !detector::detect(clear_blue, ENEMY_BLUE).empty(), "Blue plate lost");
            cv::VideoCapture cap(std::string(argv[1]) + "/assets/video/video_1.avi");
            require(cap.isOpened(), "Cannot open regression video");
            cap.set(cv::CAP_PROP_POS_FRAMES, 753);
            cv::Mat frame;
            require(cap.read(frame), "Cannot read regression frame");
            auto mask = detector::extract_color(frame, ENEMY_RED, 70, 170);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            fitted = detector::get_valid_light_rects(contours, quality, 55, 1.5, 40);
            selected = detector::match_armors(fitted, quality, 20, 2, .8f, .8f);
            bool correct = false;
            for (const auto &a : selected) {
                require(!(left_center(a).x > 700 && right_center(a).x > 800),
                        "Regression: frame 753 cross-plate pairing");
                if (std::abs(left_center(a).x - 637) < 10 &&
                    std::abs(right_center(a).x - 714) < 10) correct = true;
            }
            require(correct, "Frame 753 true plate missing");

            cap.open(std::string(argv[1]) + "/assets/video/video_2.avi");
            require(cap.isOpened(), "Cannot open side-plate regression video");
            cap.set(cv::CAP_PROP_POS_FRAMES, 41);
            require(cap.read(frame), "Cannot read side-plate regression frame");
            mask = detector::extract_color(frame, ENEMY_RED, 70, 170);
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            fitted = detector::get_valid_light_rects(contours, quality, 55, 1.5, 40);
            selected = detector::match_armors(fitted, quality, 20, 2, .8f, .8f);
            require(selected.size() == 2, "Thin side plate lost by shape filtering");

            cap.set(cv::CAP_PROP_POS_FRAMES, 220);
            require(cap.read(frame), "Cannot read refinement regression frame");
            mask = detector::extract_color(frame, ENEMY_RED, 70, 170);
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            fitted = detector::get_valid_light_rects(contours, quality, 55, 1.5, 40);
            selected = detector::match_armors(fitted,quality,20,2,.8f,.8f,3.1f,.35f);
            cv::Mat K2 = (cv::Mat_<double>(3,3) << 1711.311186,0,732.488057,
                0,1714.616882,546.930868,0,0,1);
            cv::Mat D2 = (cv::Mat_<double>(1,5) << -.119922,-.078593,.007511,-.028028,0);
            ::solver solver2{K2, D2};
            bool recovered = false;
            for (auto a : selected) {
                bool solved = solver2.solve(a);
                recovered |= solved;
            }
            require(recovered, "Endpoint refinement lost video 2 frame 220");
        }
        std::cout << "Detector and pose regression checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
