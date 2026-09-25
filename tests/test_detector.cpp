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

int main(int argc, char **argv) {
    try {
        // Frame 753 geometry: the true digit-6 pair has 15 degrees of axis skew;
        // the nearly parallel cross-plate pair is much too wide for this model.
        std::vector<cv::RotatedRect> bars{
            {{513.8f,725.8f},{37.1f,6.3f},104.f},
            {{637.f,716.f},{32.f,10.f},90.f},
            {{713.7f,715.3f},{31.5f,15.3f},74.7f},
            {{821.4f,722.5f},{29.8f,5.3f},76.f}};
        auto selected = matchArmors(bars, 20, 2, .8f, .8f);
        require(selected.size() == 1, "Expected one consistent pair");
        require(std::abs(selected[0].left_light.center.x - 637.f) < 1,
                "Wrong left light for frame-753 geometry");
        require(std::abs(selected[0].right_light.center.x - 713.7f) < 1,
                "Wrong right light for frame-753 geometry");
        auto expected = selected[0].center;
        std::reverse(bars.begin(), bars.end());
        selected = matchArmors(bars, 20, 2, .8f, .8f);
        require(cv::norm(selected[0].center - expected) < 1e-5, "Input-order dependence");

        // Four lights: select two valid plates without sharing any light.
        bars.clear();
        for (float x : {0.f, 72.f, 150.f, 222.f})
            bars.emplace_back(cv::Point2f(x,100), cv::Size2f(30,4), 90.f);
        selected = matchArmors(bars);
        require(selected.size() == 2, "Expected two non-conflicting plates");
        require(selected[0].right_light.center.x == 72.f &&
                selected[1].left_light.center.x == 150.f, "Incorrect global assignment");
        require(matchArmors({}).empty(), "Empty input");
        require(matchArmors(bars, 0, 2, .8f, .8f).empty(), "Invalid threshold");

        // First-fit would consume lights 0/1; quality matching must prefer 1/2.
        bars = {{{0,120},{30,4},90}, {{72,100},{30,4},90}, {{144,100},{30,4},90}};
        selected = matchArmors(bars);
        require(selected.size() == 1 && selected[0].left_light.center.x == 72,
                "First-fit won over the higher quality pair");

        // A short reflection spur must not pull the light center toward its tip.
        std::vector<std::vector<cv::Point>> outlines{{{97,80},{103,80},
            {103,98},{108,100},{103,102},{103,120},{97,120}}};
        std::vector<float> quality;
        auto fitted = getValidLightRects(outlines, 55, &quality);
        require(fitted.size() == 1 && quality.size() == 1, "Missing fitted light");
        require(std::abs(fitted[0].center.x-100) < 1,
                "Reflection spur biased the fitted center");
        require(std::abs(std::max(fitted[0].size.width, fitted[0].size.height)-40) < 1,
                "Light length was shortened, biasing PnP depth");
        require(quality[0] > 0 && quality[0] < 1, "Irregular contour needs a quality penalty");
        fitted = getValidLightRects({{}, {{1,1}}}, 55, &quality);
        require(fitted.empty() && quality.empty(), "Degenerate contours not skipped");

        // Optional quality must follow each light even if input order changes.
        bars = {{{0,100},{30,4},90}, {{72,100},{30,4},90}, {{144,100},{30,4},90}};
        selected = matchArmors(bars,20,1.5f,1.2f,.8f,3.1f,.35f,{.3f,1.f,1.f});
        require(selected.size() == 1 && selected[0].left_light.center.x == 72,
                "Light quality did not affect ambiguous pairing");
        std::reverse(bars.begin(), bars.end());
        selected = matchArmors(bars,20,1.5f,1.2f,.8f,3.1f,.35f,{1.f,1.f,.3f});
        require(selected.size() == 1 && selected[0].left_light.center.x == 72,
                "Light quality detached from sorted geometry");

        // A physically exact projected rectangle must survive the PnP gate.
        cv::Mat K = (cv::Mat_<double>(3,3) << 1000,0,640,0,1000,480,0,0,1);
        cv::Mat distortion = cv::Mat::zeros(1,5,CV_64F);
        std::vector<cv::Point3f> object{{-.0675f,-.028f,0}, {-.0675f,.028f,0},
                                      {.0675f,.028f,0}, {.0675f,-.028f,0}};
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object, cv::Vec3d(0,.4,0), cv::Vec3d(.1,.2,3), K, distortion, projected);
        Armor armor;
        std::copy(projected.begin(), projected.end(), armor.vertices);
        Solver solver(K, distortion);
        require(solver.solve(armor), "Rejected exact physical rectangle");
        require(armor.reprojection_error < .01, "Unexpected reprojection error");
        require(std::abs(armor.yaw+ .4*180/CV_PI) < .01, "PnP yaw sign changed");
        require(armor.pnp_candidate_count >= 1, "No validated PnP candidates recorded");
        // Distorted, tilted plates on both sides of zero must keep their pose.
        cv::Mat lens = (cv::Mat_<double>(1,5) << -.15,.03,.001,-.002,0);
        Solver distorted_solver(K, lens);
        for (double yaw : {-.9, -.3, .001, .3, .9}) {
            cv::Vec3d rotation(.08, yaw, -.03), position(.15,.1,2.5);
            cv::projectPoints(object, rotation, position, K, lens, projected);
            Armor sample;
            std::copy(projected.begin(), projected.end(), sample.vertices);
            require(distorted_solver.solve(sample), "Exact distorted plate rejected");
            cv::Mat expected_rotation, recovered_rotation;
            cv::Rodrigues(rotation, expected_rotation);
            cv::Rodrigues(sample.rvec, recovered_rotation);
            require(cv::norm(expected_rotation-recovered_rotation) < .002,
                    "Wrong planar pose branch for exact synthetic plate");
            require(cv::norm(sample.tvec-cv::Mat(position)) < .001, "PnP translation changed");
        }

        // Temporal hints use unique geometric matches, not per-frame x ordering.
        Armor first = armor, second = armor;
        first.center = {100,100}; second.center = {300,100};
        first.yaw = -20; second.yaw = 30;
        for (auto *a : {&first, &second}) {
            a->vertices[0] = a->center+cv::Point2f(-40,-15);
            a->vertices[1] = a->center+cv::Point2f(-40,15);
            a->vertices[2] = a->center+cv::Point2f(40,15);
            a->vertices[3] = a->center+cv::Point2f(40,-15);
        }
        solver.finishFrame({first,second}, 0);
        auto hints = solver.yawHints({second,first}, 1./30);
        require(hints[0] == 30 && hints[1] == -20, "Temporal matching depends on order");
        hints = solver.yawHints({first,first}, 1./30);
        require(!std::isfinite(hints[0]) && !std::isfinite(hints[1]), "Ambiguous match reused prior");
        hints = solver.yawHints({first}, .2);
        require(!std::isfinite(hints[0]), "Stale pose used after long time gap");
        solver.finishFrame({}, 1./30);
        hints = solver.yawHints({first}, 2./30);
        require(!std::isfinite(hints[0]), "Missing frame failed to clear pose history");
        armor.vertices[1] = armor.vertices[0];
        require(!solver.solve(armor), "Accepted degenerate vertices");

        if (argc > 1) {
            cv::VideoCapture cap(std::string(argv[1]) + "/assets/video/video_1.avi");
            require(cap.isOpened(), "Cannot open regression video");
            cap.set(cv::CAP_PROP_POS_FRAMES, 753);
            cv::Mat frame;
            require(cap.read(frame), "Cannot read regression frame");
            auto mask = extractColor(frame, EnemyColor::RED, 70, 170);
            auto contours = filterLightBars(extractContours(mask), 1.5, 40);
            selected = matchArmors(getValidLightRects(contours, 55), 20, 2, .8f, .8f);
            bool correct = false;
            for (const auto &a : selected) {
                require(!(a.left_light.center.x > 700 && a.right_light.center.x > 800),
                        "Regression: frame 753 cross-plate pairing");
                if (std::abs(a.left_light.center.x - 637) < 10 &&
                    std::abs(a.right_light.center.x - 714) < 10) correct = true;
            }
            require(correct, "Frame 753 true plate missing");

            cap.open(std::string(argv[1]) + "/assets/video/video_2.avi");
            require(cap.isOpened(), "Cannot open side-plate regression video");
            cap.set(cv::CAP_PROP_POS_FRAMES, 41);
            require(cap.read(frame), "Cannot read side-plate regression frame");
            mask = extractColor(frame, EnemyColor::RED, 70, 170);
            contours = filterLightBars(extractContours(mask), 1.5, 40);
            selected = matchArmors(getValidLightRects(contours, 55), 20, 2, .8f, .8f);
            require(selected.size() == 2, "Thin side plate lost by shape filtering");

            // Endpoint refinement must not turn a recoverable PnP failure into
            // a missing frame. Fallback reuses the same lights and same solver.
            cap.set(cv::CAP_PROP_POS_FRAMES, 220);
            require(cap.read(frame), "Cannot read refinement fallback frame");
            mask = extractColor(frame, EnemyColor::RED, 70, 170);
            contours = filterLightBars(extractContours(mask), 1.5, 40);
            std::vector<cv::RotatedRect> originals;
            fitted = getValidLightRects(contours, 55, &quality, &originals);
            selected = matchArmors(fitted,20,2,.8f,.8f,3.1f,.35f,quality);
            cv::Mat K2 = (cv::Mat_<double>(3,3) << 1711.311186,0,732.488057,
                0,1714.616882,546.930868,0,0,1);
            cv::Mat D2 = (cv::Mat_<double>(1,5) << -.119922,-.078593,.007511,-.028028,0);
            Solver solver2(K2, D2);
            bool recovered = false;
            for (auto a : selected) {
                bool solved = solver2.solve(a);
                if (!solved && restoreLightEndpoints(a, fitted, originals))
                    solved = solver2.solve(a);
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
