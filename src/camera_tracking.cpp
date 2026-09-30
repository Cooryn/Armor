#include "camera_tracking.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace {
bool project(const std::vector<cv::Point3d> &points, const cv::Mat &camera,
             const cv::Mat &distortion, std::vector<cv::Point> &pixels) {
    for (const auto &p : points)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z <= .1)
            return false;
    std::vector<cv::Point2d> projected;
    cv::projectPoints(points, cv::Vec3d(), cv::Vec3d(), camera, distortion, projected);
    for (const auto &p : projected) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
        pixels.emplace_back(cvRound(std::clamp(p.x, -1e6, 1e6)), cvRound(std::clamp(p.y, -1e6, 1e6)));
    }
    return true;
}
void dashed(cv::Mat &image, cv::Point a, cv::Point b, const cv::Scalar &color) {
    if (!cv::clipLine(image.size(), a, b)) return;
    const double length = cv::norm(b-a);
    if (length < 1) return;
    const cv::Point2d direction = cv::Point2d(b-a) / length;
    for (double offset = 0; offset < length; offset += 10)
        cv::line(image, cv::Point2d(a) + direction*offset,
                 cv::Point2d(a) + direction*std::min(offset+6, length), color, 2, cv::LINE_AA);
}
}


void CameraTracking::draw(cv::Mat &image, const cv::Mat &camera, const cv::Mat &distortion, int frame_id) {
    const int width=image.cols, height=image.rows;
    cv::Mat canvas(height,width+340,CV_8UC3,cv::Scalar(30,27,24));
    image.copyTo(canvas(cv::Rect(0,0,width,height)));
    cv::Mat view=canvas(cv::Rect(0,0,width,height));
    const cv::Scalar colors[]={{90,100,255},{255,180,40},{80,230,110},{20,190,255}};
    const cv::Rect bounds(0,0,width,height);
    std::vector<cv::Point> detail;
    std::optional<cv::Point> current_center;
    std::optional<cv::Point> plate_centers[4];
    auto pixel = [&](const cv::Point3d &point)->std::optional<cv::Point> {
        std::vector<cv::Point> result;
        if (!project({point},camera,distortion,result) || !bounds.contains(result[0])) return std::nullopt;
        detail.push_back(result[0]);
        return result[0];
    };
    auto plates = [&](const predictor::Forecast &forecast, bool future) {
        for(int id=0; id<4; ++id) {
            const auto &p=forecast.plates;
            const cv::Point3d center(p(id,0),p(id,1),p(id,2));
            const cv::Point3d w(.0675*std::cos(p(id,3)),0,.0675*std::sin(p(id,3))), h(0,.028,0);
            std::vector<cv::Point> corners;
            if(project({center-w-h,center-w+h,center+w+h,center+w-h},camera,distortion,corners)) {
                for(int j=0;j<4;++j) {
                    auto a=corners[j], b=corners[(j+1)%4];
                    if(!cv::clipLine(bounds,a,b)) continue;
                    detail.push_back(a); detail.push_back(b);
                    if(future) dashed(view,a,b,colors[id]);
                    else cv::line(view,a,b,colors[id],2,cv::LINE_AA);
                }
            }
            if(auto point=pixel(center)) {
                if(future) cv::drawMarker(view,*point,colors[id],cv::MARKER_DIAMOND,11,1,cv::LINE_AA);
                else {
                    plate_centers[id]=point;
                    cv::circle(view,*point,8,colors[id],2,cv::LINE_AA);
                    if(current_center) cv::line(view,*current_center,*point,colors[id],1,cv::LINE_AA);
                }
                cv::putText(view,std::string(future?"F":"A")+std::to_string(id),
                    *point+cv::Point(future?8:11,future?16:-9),cv::FONT_HERSHEY_SIMPLEX,
                    future?.45:.5,colors[id],1,cv::LINE_AA);
            }
        }
    };
    if(visible()) {
        const auto current=ekf_.forecast(0), future=ekf_.forecast(horizon_s);
        current_center=pixel({ekf_.X(0),ekf_.X(2),ekf_.X(4)});
        if(current_center) {
            if(drawn_time_!=last_time_) {
                trail_.push_back(*current_center);
                if(trail_.size()>30) trail_.pop_front();
            }
            for(size_t i=1;i<trail_.size();++i) cv::line(view,trail_[i-1],trail_[i],{180,180,180},1,cv::LINE_AA);
            cv::drawMarker(view,*current_center,{255,255,255},cv::MARKER_STAR,15,2);
        }
        plates(current,false);
        if(auto point=pixel({future.state(0),future.state(2),future.state(4)}))
            cv::drawMarker(view,*point,{255,255,255},cv::MARKER_DIAMOND,13,1);
        plates(future,true);
    } else trail_.clear();
    drawn_time_=last_time_;
    constexpr double rad = 3.14159265358979323846 / 180.;
    const auto axis = (Eigen::AngleAxisd(simulated_angles_.x()*rad,Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(simulated_angles_.y()*rad,Eigen::Vector3d::UnitX())) * Eigen::Vector3d::UnitZ();
    std::vector<cv::Point> optical_axis;
    if (project({{axis.x(),axis.y(),axis.z()}},camera,distortion,optical_axis) && bounds.contains(optical_axis[0])) {
        cv::drawMarker(view,optical_axis[0],{255,255,0},cv::MARKER_CROSS,24,2);
        cv::putText(view,"SIM CAMERA AXIS",optical_axis[0]+cv::Point(12,-12),cv::FONT_HERSHEY_SIMPLEX,.5,{255,255,0},1,cv::LINE_AA);
    }
    if (mode_==Mode::Automatic && command_.target_valid) {
        std::vector<cv::Point> target;
        if(project({{predicted_target_.x(),predicted_target_.y(),predicted_target_.z()}},camera,distortion,target) && bounds.contains(target[0]))
            cv::drawMarker(view,target[0],{255,0,255},cv::MARKER_TILTED_CROSS,20,2);
    }
    size_t rejected=0, unknown=0;
    for(size_t i=0;i<observations_.size();++i) {
        const auto &d=diagnostics_[i]; const auto &z=observations_[i].Z_obs;
        const bool reject=d.reason=="invalid" || d.reason=="innovation_gate" || d.reason=="association_conflict";
        rejected+=reject; unknown+=!d.accepted && !reject;
        if(z.rows()!=4 || z.cols()!=1 || !z.allFinite() || z(2)<=0) continue;
        const auto point=pixel({z(2)*std::cos(z(1))*std::sin(z(0)),z(2)*std::sin(z(1)),z(2)*std::cos(z(1))*std::cos(z(0))});
        if(!point) continue;
        const auto color=d.accepted ? colors[d.armor_id] : (reject?cv::Scalar(40,80,255):cv::Scalar(0,220,255));
        cv::drawMarker(view,*point,color,reject?cv::MARKER_TILTED_CROSS:cv::MARKER_CROSS,18,2,cv::LINE_AA);
        if(d.accepted && plate_centers[d.armor_id]) cv::line(view,*point,*plate_centers[d.armor_id],color,1,cv::LINE_AA);
    }
    using Line=std::pair<std::string,cv::Scalar>;
    const cv::Scalar white(235,235,235), gray(210,210,210);
    std::vector<Line> lines={{"ARMOR TRACKING",white},
        {cv::format("Frame %d | %.2f s",frame_id,last_time_.value_or(0)/1000),gray},
        {status_=="tracking"?"UPDATED":status_=="prediction_only"?"PREDICTION ONLY":status_=="lost"?"LOST":"INITIALIZING / NO STATE",
            status_=="tracking"?cv::Scalar(80,220,100):cv::Scalar(0,200,255)},
        {"ESTIMATE (camera-frame EKF)",gray}};
    if(visible()) {
        const auto &x=ekf_.X;
        lines.push_back({cv::format("xc: %.3f m",x(0)),white});
        lines.push_back({cv::format("yc: %.3f m",x(2)),white});
        lines.push_back({cv::format("zc: %.3f m",x(4)),white});
        lines.push_back({cv::format("body_yaw: %.3f rad",x(6)),white});
        lines.push_back({cv::format("w: %.2f rad/s",x(7)),white});
        lines.push_back({cv::format("r: %.3f m",x(8)),white});
        lines.push_back({cv::format("dl / dh: %.3f / %.3f",x(9),x(10)),white});
        lines.push_back({"FORECAST: +50 ms",white});
        lines.push_back({cv::format("Target time: %.3f s",last_time_.value_or(0)/1000+horizon_s),gray});
    }
    lines.insert(lines.end(),{{"",gray},{"Observations: "+std::to_string(observations_.size()),white},
        {"Accepted: "+std::to_string(accepted_)+"   Rejected: "+std::to_string(rejected),white},
        {"Unclassified: "+std::to_string(unknown),{0,220,255}},
        {"LEGEND",white},{"Solid A0-A3: current estimate",gray},{"Dashed F0-F3: future forecast",gray},
        {"+ : observation",gray},{"Red X: rejected observation",{40,80,255}},
        {"Yellow +: not classified",{0,220,255}},{"Star / trail: center",gray},
        {"Diamond: future center/plate",gray},{"A/F pairs: same relative ID",gray},
        {"CAMERA TRACKING (preview)",white}});
    if(command_.target_valid) lines.push_back({cv::format("Yaw / pitch: %.2f / %.2f deg",command_.yaw_error_deg,command_.pitch_error_deg),gray});
    else lines.push_back({"Yaw / pitch: unavailable",gray});
    lines.push_back({cv::format("Rate: %.2f / %.2f deg/s",command_.yaw_rate_dps,command_.pitch_rate_dps),gray});
    lines.push_back({command_.control_valid?"VALID | no motor commands":"HOLD | no motor commands",{0,220,255}});
    lines.insert(lines.begin(), {
        {std::string("SIMULATION | ")+mode_name(), {0,220,255}},
        {"1 Manual | 2 Auto | 3/C Center", white},
        {"WASD: manual target +/-2 deg", gray},
        {cv::format("Sim yaw/pitch: %.1f / %.1f",simulated_angles_.x(),simulated_angles_.y()),white}});
    const int bottom=height>=1000?height-240:height-10;
    const int spacing=std::min(27,std::max(10,(bottom-32)/int(lines.size())));
    for(size_t i=0;i<lines.size();++i) {
        const int y=32+int(i)*spacing; if(y>=bottom) break;
        cv::putText(canvas,lines[i].first,{width+16,y},cv::FONT_HERSHEY_SIMPLEX,std::min(.53,spacing/30.),lines[i].second,1,cv::LINE_AA);
    }
    if(!detail.empty() && height>=1000) {
        cv::Rect area=cv::boundingRect(detail);
        const cv::Point2d middle(area.x+area.width/2.,area.y+area.height/2.);
        const int w=std::max(200,area.width+70), h=std::max(140,area.height+70);
        area=cv::Rect(int(middle.x-w/2),int(middle.y-h/2),w,h)&bounds;
        if(area.area()>0) {
            const double scale=std::min(308./area.width,180./area.height);
            cv::Mat zoom; cv::resize(view(area),zoom,{std::max(1,int(area.width*scale)),std::max(1,int(area.height*scale))});
            zoom.copyTo(canvas(cv::Rect(width+16+(308-zoom.cols)/2,height-195,zoom.cols,zoom.rows)));
            cv::putText(canvas,"TARGET DETAIL",{width+16,height-210},cv::FONT_HERSHEY_SIMPLEX,.5,gray,1,cv::LINE_AA);
        }
    }
    image=canvas;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <cctype>
#include <algorithm>
#include <fstream>
#include <cmath> // 引入 cmath 以使用 std::remainder
#include <iomanip>
#include <chrono>

#include "lightbar_detector.hpp"
#include "solver.hpp"
#include "camera_tracking.hpp"

namespace fs = std::filesystem;

int export_control_csv(int argc, char **argv);

int main(int argc, char **argv)
{
    if (argc > 1 && std::string(argv[1]) == "--input") return export_control_csv(argc, argv);
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [red|blue] [video file] [--headless]\n"
                  << "Live preview at source FPS; Esc or close window to stop. --headless disables preview.\n"
                  << "Defaults: red video_1.avi. Keys: 1 manual, 2 auto, 3/C center; WASD manual steps.\n";
        std::cout << "Outputs: results/camera_prediction_*.avi and data/camera_pose_raw_*.csv\n";
        return 0;
    }
    if (argc > 4 || (argc == 4 && std::string(argv[3]) != "--headless")) {
        std::cerr << "Usage: " << argv[0] << " [red|blue] [video file] [--headless]" << std::endl;
        return 1;
    }
    const bool preview = argc < 4;
    std::string color = argc >= 2 ? argv[1] : "red";
    std::transform(color.begin(), color.end(), color.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (color != "red" && color != "blue") {
        std::cerr << "Color must be red or blue" << std::endl;
        return 1;
    }
    const EnemyColor target_color = color == "red" ? EnemyColor::RED : EnemyColor::BLUE;
    const std::string input_filename = argc >= 3 ? argv[2] : "video_1.avi";
    std::string input_path = input_filename;
    if (!fs::exists(input_path)) input_path = (fs::path("assets/video") / input_filename).string();
    std::string extension = fs::path(input_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension != ".avi" && extension != ".mp4" && extension != ".mov" &&
        extension != ".mkv" && extension != ".m4v" && extension != ".webm" &&
        extension != ".mpg" && extension != ".mpeg" && extension != ".wmv") {
        std::cerr << "Unsupported video file extension: " << extension << std::endl;
        return 1;
    }
    cv::VideoCapture stream(input_path);
    cv::Mat original_frame;
    if (!stream.isOpened() || !stream.read(original_frame)) {
        std::cerr << "Cannot read video: " << input_path << std::endl;
        return 1;
    }
    const double source_fps = stream.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(source_fps) || source_fps <= 0) {
        std::cerr << "Invalid source video FPS" << std::endl;
        return 1;
    }
    const char *window = "Camera prediction - Esc to stop";
    if (preview) {
        try {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            cv::resizeWindow(window, 960, 720);
        } catch (const cv::Exception &error) {
            std::cerr << "Cannot create OpenCV window; use --headless. " << error.what() << std::endl;
            return 1;
        }
    }
    constexpr int color_th = 70, gray_th = 170, min_area = 40;
    constexpr float min_angle = 55, max_angle_diff = 20;
    std::string stem = fs::path(input_path).stem().string();
    std::string out_dir = "./results/";
    fs::create_directories(out_dir);

    std::string output_path = out_dir + "camera_prediction_" + stem + ".avi";

    cv::VideoWriter writer;
    std::ofstream csv_file;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::Size output_size = original_frame.size();
    output_size.width += 340;
    writer.open(output_path, fourcc, source_fps, output_size, true);
    if (!writer.isOpened()) { std::cerr << "Cannot open output video" << std::endl; return 1; }

    const auto separator = stem.find_last_of('_');
    std::string suffix = separator == std::string::npos ? "_" + stem : stem.substr(separator);
    std::string csv_filename = "camera_pose_raw" + suffix + ".csv";

    // 确保 ./data/ 目录存在
    fs::create_directories("./data/");
    csv_file.open("./data/" + csv_filename);
    if (!csv_file.is_open()) { std::cerr << "Cannot open output CSV" << std::endl; return 1; }
    csv_file << std::setprecision(15);
    csv_file << "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,detection_score,reprojection_error,pnp_candidate_count,pnp_used_temporal,rvec_x,rvec_y,rvec_z,coordinate_frame\n";

    cv::Mat camera_matrix, distort_coeffs;

    if (stem.find("video_2") != std::string::npos)
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0.000000, 732.488057,
                         0.000000, 1714.616882, 546.930868,
                         0.000000, 0.000000, 1.000000);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.119922, -0.078593, 0.007511, -0.028028, 0.000000);
    }
    else
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                         0, 1288.1400736562441, 483.6163720308021,
                         0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
    }

    Solver pnp_solver(camera_matrix, distort_coeffs);

    CameraTracking tracking;
    std::ofstream controls(out_dir + "camera_control" + suffix + ".csv");
    if (!controls) { std::cerr << "Cannot open control CSV\n"; return 1; }
    controls << std::setprecision(15);
    controls << "frame_id,timestamp_ms,mode,simulation_only,target_valid,yaw_deg,pitch_deg,yaw_rate_dps,pitch_rate_dps,control_valid,target_x,target_y,target_z\n";
    int frame_count = 0;

    // Frame timestamps use the source video clock.

    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        cv::Mat frame = original_frame.clone();

        cv::Mat mask = extractColor(frame, target_color, color_th, gray_th);
        auto contours = extractContours(mask);
        std::vector<float> light_quality;
        auto lightRects = getValidLightRects(contours, (float)min_angle, &light_quality, 1.5, min_area);
        auto armors = matchArmors(lightRects, max_angle_diff, 2.f, .8f, .8f, 3.1f, .35f, light_quality);

        cv::Mat final_result = frame.clone();

        int text_y_offset = 30;
        std::vector<Armor> valid_armors;
        const double source_time = frame_count/source_fps;
        const auto yaw_hints = pnp_solver.yawHints(armors, source_time);

        for (size_t i = 0; i < armors.size(); i++)
        {
            bool solved = pnp_solver.solve(armors[i], yaw_hints[i]);
            if (solved)
            {
                valid_armors.push_back(armors[i]);

                double tx = armors[i].tvec.at<double>(0), ty = armors[i].tvec.at<double>(1), tz = armors[i].tvec.at<double>(2);
                double rx = armors[i].rvec.at<double>(0), ry = armors[i].rvec.at<double>(1), rz = armors[i].rvec.at<double>(2);

                cv::putText(final_result, cv::format("Armor[%zu] tvec: x %5.2f y %5.2f z %5.2f", i, tx, ty, tz),
                            cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 25;

                cv::putText(final_result, cv::format("Armor[%zu] rvec: x %5.2f y %5.2f z %5.2f", i, rx, ry, rz),
                            cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 30;
            }
        }

        pnp_solver.finishFrame(valid_armors, source_time);
        drawArmors(final_result, valid_armors);
        std::vector<predictor::Observation> observations;
        if (!valid_armors.empty())
        {
            // Source video time in milliseconds.
            double timestamp = frame_count * 1000.0 / source_fps;

            // 🚀 核心修改：不再只取 [0]，而是遍历所有合法的装甲板
            for (size_t i = 0; i < valid_armors.size(); i++)
            {
                Armor current_armor = valid_armors[i];

                // 1. 获取 PnP 算出的原始 x, y, z
                double tx = current_armor.tvec.at<double>(0);
                double ty = current_armor.tvec.at<double>(1);
                double tz = current_armor.tvec.at<double>(2);
                double armor_yaw_rad = current_armor.yaw * CV_PI / 180.0;

                // 2. 解算纯几何维度的目标朝向
                double target_yaw = std::atan2(tx, tz);
                double target_pitch = std::atan2(ty, std::sqrt(tx * tx + tz * tz));
                double distance = std::sqrt(tx * tx + ty * ty + tz * tz);

                // 将角度约束在 [-PI, PI]
                double armor_orientation_yaw = std::remainder(armor_yaw_rad, 2.0 * CV_PI);
                observations.push_back({predictor::ArmorEKF::make_vector(
                    {target_yaw, target_pitch, distance, armor_orientation_yaw})});

                // 3. 写入 CSV (同一个 frame_count 会被写入多次，占多行)
                csv_file << frame_count << ","
                         << timestamp << ","
                         << tx << ","
                         << ty << ","
                         << tz << ","
                         << target_yaw << ","
                         << target_pitch << ","
                         << distance << ","
                         << armor_orientation_yaw << ","
                         << current_armor.detection_score << ","
                         << current_armor.reprojection_error << ","
                         << current_armor.pnp_candidate_count << ","
                         << current_armor.pnp_used_temporal << ","
                         << current_armor.rvec.at<double>(0) << ","
                         << current_armor.rvec.at<double>(1) << ","
                         << current_armor.rvec.at<double>(2) << ",camera\n";

            }
        }

        tracking.update(observations, frame_count * 1000.0 / source_fps);
        final_result = frame.clone();
        tracking.draw(final_result, camera_matrix, distort_coeffs, frame_count);
        const auto &command = tracking.command();
        const auto &angles = tracking.simulated_angles();
        const auto &target = tracking.predicted_target();
        controls << frame_count << ',' << frame_count*1000.0/source_fps << ',' << tracking.mode_name()
                 << ",1," << (tracking.mode()==CameraTracking::Mode::Automatic && command.target_valid) << ',' << angles.x() << ',' << angles.y() << ',' << command.yaw_rate_dps << ','
                 << command.pitch_rate_dps << ',' << command.control_valid << ','
                 << target.x() << ',' << target.y() << ',' << target.z() << '\n';
        writer.write(final_result);
        ++frame_count;
        if (preview) {
            cv::imshow(window, final_result);
            const auto deadline = frame_start + std::chrono::duration<double>(1.0 / source_fps);
            bool stop = false;
            do {
                const double remaining_ms = std::chrono::duration<double, std::milli>(
                    deadline - std::chrono::steady_clock::now()).count();
                const int delay = static_cast<int>(std::clamp(std::ceil(remaining_ms), 1.0, 10.0));
                const int key = cv::waitKey(delay) & 0xff;
                tracking.key(key);
                stop = key == 27;
                if (!stop) {
                    // Qt destroys its receiver when the last window is closed.
                    try { stop = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1; }
                    catch (const cv::Exception &) { stop = true; }
                }
                if (stop) break;
            } while (std::chrono::steady_clock::now() < deadline);
            if (stop) break;
        }
        if (!stream.read(original_frame)) break;

    } while (true);

    writer.release();
    csv_file.close();
    if (preview) cv::destroyAllWindows();

    std::cout << "视频已导出至 " << out_dir << " (共处理 " << frame_count << " 帧)" << std::endl;

    return 0;
}

#include "camera_tracking.hpp"
#include "camera_frames.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

namespace {
namespace fs = std::filesystem;
inline std::vector<std::vector<std::string>> read_csv(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read " + p.string());
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool in_quote = false;
    char c;
    while (f.get(c)) {
        if (c == '\"') {
            if (in_quote && f.peek() == '\"') {
                f.get(c);
                field += '\"';
            } else
                in_quote = !in_quote;
        } else if (!in_quote && (c == ',' || c == '\n' || c == '\r')) {
            row.push_back(field);
            field.clear();
            if (c != ',') {
                if (c == '\r' && f.peek() == '\n')
                    f.get(c);
                if (row.size() > 1 || !row[0].empty())
                    rows.push_back(row);
                row.clear();
            }
        } else
            field += c;
    }
    if (in_quote)
        throw std::invalid_argument("Unclosed CSV quote");
    if (!field.empty() || !row.empty()) {
        row.push_back(field);
        rows.push_back(row);
    }
    if (rows.empty())
        throw std::invalid_argument("No columns to parse from file");
    if (rows[0][0].compare(0, 3, "\xef\xbb\xbf") == 0)
        rows[0][0].erase(0, 3);
    return rows;
}
double numeric(const std::string &value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::numeric_limits<double>::quiet_NaN();
    const std::string text = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
    size_t used = 0;
    const double result = std::stod(text, &used);
    if (used != text.size())
        throw std::invalid_argument("Invalid numeric value: " + text);
    return result;
}
struct Record {
    double frame, timestamp;
    CameraGimbal::Output command;
};
void export_commands(const fs::path &input, const fs::path &output,
                     const CameraGimbal::Config &config, const std::optional<CameraFrames> &frames) {
    CameraGimbal controller(config);
    const auto csv = read_csv(input);
    std::map<std::string, size_t> columns;
    for (size_t i = 0; i < csv.front().size(); ++i)
        if (!columns.emplace(csv.front()[i], i).second)
            throw std::invalid_argument("Duplicate CSV column");
    for (const char *key : {"frame_id", "timestamp", "xc", "yc", "zc", "accepted_count", "status"})
        if (!columns.count(key))
            throw std::invalid_argument(std::string("Missing CSV column: ") + key);
    std::vector<Record> records;
    double previous_frame = -1;
    std::string source_frame;
    for (size_t i = 1; i < csv.size(); ++i) {
        const auto &row = csv[i];
        if (row.size() != csv.front().size())
            throw std::invalid_argument("CSV row width mismatch at row " + std::to_string(i + 1));
        auto get = [&](const std::string &key) { return numeric(row.at(columns.at(key))); };
        const double frame = get("frame_id"), timestamp = get("timestamp"), accepted = get("accepted_count");
        if (!std::isfinite(frame) || frame < 0 || std::floor(frame) != frame || frame <= previous_frame)
            throw std::invalid_argument("Frame IDs must be increasing nonnegative integers");
        if (!std::isfinite(accepted) || accepted < 0 || std::floor(accepted) != accepted)
            throw std::invalid_argument("Invalid accepted_count");
        const auto &status = row.at(columns.at("status"));
        if ((status != "updated" && status != "prediction_only") ||
            (status == "updated") != (accepted > 0))
            throw std::invalid_argument("Inconsistent tracking status and accepted_count");
        const std::string frame_name = columns.count("coordinate_frame") ? row.at(columns.at("coordinate_frame")) : "camera";
        if ((frame_name != "camera" && frame_name != "base") ||
            (!source_frame.empty() && source_frame != frame_name))
            throw std::invalid_argument("Unknown or mixed input coordinate frames");
        source_frame = frame_name;
        if ((frame_name == "base") != frames.has_value())
            throw std::invalid_argument("Base-frame input requires telemetry/calibration; camera-frame input must omit them");
        Eigen::Vector3d center(get("xc"), get("yc"), get("zc"));
        if (frames) {
            const double reference = get("base_reference_timestamp_ms");
            const Eigen::Vector3d origin(get("base_origin_x_m"), get("base_origin_y_m"), get("base_origin_z_m"));
            if (!std::isfinite(reference) || !origin.allFinite() ||
                std::abs(reference - frames->reference_time_ms()) > 1e-6 ||
                (origin - frames->origin_in_mechanical()).norm() > 1e-9)
                throw std::invalid_argument("Base origin mismatch: use the same --origin-time-ms as pose_base");
            center = frames->at(timestamp).inverse() * center;
        }
        auto command = controller.update(timestamp, center,
                                          status == "updated");
        records.push_back({frame, timestamp, command});
        previous_frame = frame;
    }
    // Validate the whole input before replacing an existing command CSV.
    if (!output.parent_path().empty())
        fs::create_directories(output.parent_path());
    std::ofstream out(output);
    if (!out)
        throw std::runtime_error("Cannot write output: " + output.string());
    out << "frame_id,timestamp,yaw_error_deg,pitch_error_deg,yaw_offset_deg,pitch_offset_deg,"
           "yaw_rate_dps,pitch_rate_dps,target_valid,control_valid,angle_limited,speed_limited,"
           "acceleration_limited,status\n" << std::setprecision(17) << std::boolalpha;
    for (const auto &r : records) {
        const auto &c = r.command;
        out << r.frame << ',' << r.timestamp << ',';
        if (std::isfinite(c.yaw_error_deg)) out << c.yaw_error_deg;
        out << ',';
        if (std::isfinite(c.pitch_error_deg)) out << c.pitch_error_deg;
        out << ',' << c.yaw_offset_deg << ',' << c.pitch_offset_deg << ','
            << c.yaw_rate_dps << ',' << c.pitch_rate_dps << ',' << c.target_valid << ','
            << c.control_valid << ',' << c.angle_limited << ',' << c.speed_limited << ','
            << c.acceleration_limited << ',' << c.status << '\n';
    }
    out.close();
    if (!out)
        throw std::runtime_error("Command CSV write failed");
    std::cout << "Saved " << records.size() << " camera control rows: " << output.string() << '\n';
}
} // namespace

int export_control_csv(int argc, char **argv) {
    try {
        CameraGimbal::Config config;
        fs::path root = ".", input, output, telemetry, calibration;
        double sync_gap_ms = 100;
        std::optional<double> reference_time_ms;
        std::string suffix = "1";
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            auto value = [&]() {
                if (++i >= argc) throw std::invalid_argument("Missing value for " + argument);
                return std::string(argv[i]);
            };
            if (argument == "--suffix") suffix = value();
            else if (argument == "--root") root = fs::u8path(value());
            else if (argument == "--input") input = fs::u8path(value());
            else if (argument == "--output") output = fs::u8path(value());
            else if (argument == "--telemetry") telemetry = fs::u8path(value());
            else if (argument == "--calibration") calibration = fs::u8path(value());
            else if (argument == "--sync-gap-ms") sync_gap_ms = numeric(value());
            else if (argument == "--origin-time-ms") reference_time_ms = numeric(value());
            else if (argument == "--gain") config.gain = numeric(value());
            else if (argument == "--yaw-limit-deg") config.yaw_limit_deg = numeric(value());
            else if (argument == "--pitch-limit-deg") config.pitch_limit_deg = numeric(value());
            else if (argument == "--yaw-speed-dps") config.yaw_speed_dps = numeric(value());
            else if (argument == "--pitch-speed-dps") config.pitch_speed_dps = numeric(value());
            else if (argument == "--acceleration-dps2") config.acceleration_dps2 = numeric(value());
            else if (argument == "--deadband-deg") config.deadband_deg = numeric(value());
            else if (argument == "--max-gap-ms") config.max_gap_ms = numeric(value());
            else if (argument == "--help" || argument == "-h") {
                std::cout << "Camera-only tracking CSV exporter. Angles: right-positive yaw, up-positive pitch.\n"
                    "--suffix N --root DIR --input CSV --output CSV\n"
                    "Base-frame input: --telemetry CSV --calibration CSV --sync-gap-ms 100 [--origin-time-ms MS]\n"
                    "--gain 2 --yaw-limit-deg 45 --pitch-limit-deg 30\n"
                    "--yaw-speed-dps 60 --pitch-speed-dps 45 --acceleration-dps2 180\n"
                    "--deadband-deg 0.2 --max-gap-ms 200\n"
                    "Offsets are relative to the current optical axis, not absolute encoder angles.\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (input.empty()) input = root / "results" / ("armor_prediction_result_" + suffix + ".csv");
        if (output.empty()) output = root / "results" / ("camera_gimbal_" + suffix + ".csv");
        std::optional<CameraFrames> frames;
        if (telemetry.empty() != calibration.empty())
            throw std::invalid_argument("Specify both --telemetry and --calibration");
        if (reference_time_ms && telemetry.empty())
            throw std::invalid_argument("--origin-time-ms requires telemetry/calibration");
        if (!telemetry.empty()) frames = CameraFrames::load(telemetry, calibration, sync_gap_ms, reference_time_ms);
        for (const auto &source : {input, telemetry, calibration}) {
            if (source.empty()) continue;
            if (fs::weakly_canonical(source) == fs::weakly_canonical(output) ||
                (fs::exists(output) && fs::equivalent(source, output)))
                throw std::invalid_argument("Output must not overwrite any input file");
        }
        export_commands(input, output, config, frames);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
