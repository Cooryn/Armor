#include "lightbar_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <unordered_map>

cv::Mat extractColor(const cv::Mat &src, EnemyColor color, int color_th, int gray_th)
{
    if (src.empty() || src.channels() < 3)
        return cv::Mat::zeros(src.size(), CV_8UC1);

    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    cv::Mat color_mask;

    if (color == EnemyColor::RED)
    {
        cv::Mat r_sub_b;
        cv::subtract(channels[2], channels[0], r_sub_b);
        color_mask = (r_sub_b > color_th) & (channels[2] > gray_th);
    }
    else
    {
        cv::Mat b_sub_r;
        cv::subtract(channels[0], channels[2], b_sub_r);
        color_mask = (b_sub_r > color_th) & (channels[0] > gray_th);
    }

    cv::Mat gray, highlight_mask;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, highlight_mask, 210, 255, cv::THRESH_BINARY);

    cv::Mat shield;
    cv::Mat big_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::dilate(color_mask, shield, big_kernel);

    cv::bitwise_and(highlight_mask, shield, highlight_mask);

    cv::bitwise_or(color_mask, highlight_mask, color_mask);

    cv::Mat small_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, small_kernel);

    return color_mask;
}

std::vector<cv::RotatedRect> getValidLightRects(
    const std::vector<std::vector<cv::Point>> &lightBars, float min_angle,
    std::vector<float> *quality, std::vector<cv::RotatedRect> *original_rects)
{
    std::vector<cv::RotatedRect> rects;
    if (quality) quality->clear();
    if (original_rects) original_rects->clear();
    for (const auto &c : lightBars)
    {
        if (c.size() < 3 || cv::contourArea(c) <= 0) continue;
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        float angle = std::abs(rect.angle);
        float longEdgeAngle = (w >= h) ? angle : (90.0f - angle);

        if (longEdgeAngle < min_angle)
            continue;

        if (original_rects) original_rects->push_back(rect);
        // Resample the closed boundary uniformly. CHAIN_APPROX_SIMPLE otherwise
        // gives corners/outliers disproportionate influence over the fitted axis.
        std::vector<cv::Point2f> boundary;
        for (size_t i = 0; i < c.size(); ++i) {
            const cv::Point2f a(c[i]), b(c[(i + 1) % c.size()]);
            const int count = std::max(1, static_cast<int>(std::ceil(cv::norm(b-a))));
            for (int j = 0; j < count; ++j)
                boundary.push_back(a + (b-a) * (static_cast<float>(j) / count));
        }
        cv::Vec4f line;
        cv::fitLine(boundary, line, cv::DIST_HUBER, 0, .01, .01);
        cv::Point2f axis(line[0], line[1]);
        if (axis.y < 0) axis *= -1.f;
        const cv::Point2f origin(line[2], line[3]);
        float fitted_angle = std::atan2(axis.y, axis.x) * static_cast<float>(180.0 / CV_PI);
        const float old_angle = rect.angle + (w < h ? 90.f : 0.f);
        const float axis_change = std::abs(std::remainder(fitted_angle-old_angle, 180.f));
        float fit_quality = .75f;
        // Broad/irregular blobs do not have a reliably identifiable long axis.
        if (std::max(w,h) >= 2.f * std::min(w,h) && axis_change <= 10.f) {
            // Blend rather than replace the bounding-box axis: short rasterized
            // contours cannot support an arbitrarily precise line orientation.
            fitted_angle = old_angle + .5f * std::remainder(fitted_angle-old_angle, 180.f);
            const float radians = fitted_angle * static_cast<float>(CV_PI / 180.0);
            axis = cv::Point2f(std::cos(radians), std::sin(radians));
            if (axis.y < 0) axis *= -1.f;
            const cv::Point2f normal(axis.y, -axis.x);
            std::vector<float> along, across;
            for (const auto &point : boundary) {
                along.push_back((point-origin).dot(axis));
                across.push_back((point-origin).dot(normal));
            }
            std::sort(along.begin(), along.end());
            std::sort(across.begin(), across.end());
            // Preserve the full measured length: trimming both ends biases depth.
            const float length = along.back()-along.front();
            const float width = across.back()-across.front();
            const float offset = (across[across.size()/10] + across[across.size()*9/10]) / 2;
            const cv::Point2f center = origin + axis*((along.front()+along.back())/2)
                                      + normal*offset;
            rect = cv::RotatedRect(center, cv::Size2f(length, width), fitted_angle);
            fit_quality = 1.f;
        }
        const float fill = static_cast<float>(cv::contourArea(c)) /
                           std::max(1.f, rect.size.area());
        // A bounded confidence penalty for incomplete/irregular light shapes.
        // Oblique plate views are not penalized just for their screen angle.
        if (quality) quality->push_back(std::sqrt(fit_quality * std::clamp(fill/.75f, .5f, 1.f)));
        rects.push_back(rect);
    }
    return rects;
}

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat maskCopy = mask.clone();
    cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

std::vector<std::vector<cv::Point>> filterLightBars(const std::vector<std::vector<cv::Point>> &contours, double minAspectRatio, double minArea)
{
    std::vector<std::vector<cv::Point>> result;
    for (const auto &c : contours)
    {
        double area = cv::contourArea(c);
        if (area < minArea)
            continue;

        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        if (w < 1.0f || h < 1.0f)
            continue;

        float aspect = std::max(w, h) / std::min(w, h);
        const double fill_ratio = area / (w * h);
        if (aspect >= minAspectRatio && fill_ratio >= 0.30)
        {
            result.push_back(c);
        }
    }
    return result;
}

cv::Mat drawLightBarRects(const cv::Mat &src, const std::vector<std::vector<cv::Point>> &lightBars)
{
    cv::Mat out = src.clone();
    for (const auto &c : lightBars)
    {
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w = rect.size.width;
        float h = rect.size.height;
        float angle = std::abs(rect.angle);
        float longEdgeAngle = (w >= h) ? angle : (90.0f - angle);
        if (longEdgeAngle < 75.0f)
            continue;

        cv::Point2f vertices[4];
        rect.points(vertices);
        for (int j = 0; j < 4; j++)
        {
            cv::line(out, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 255), 2);
        }
    }
    return out;
}

namespace {
struct LightGeometry {
    float length, width;
    cv::Point2f axis;  // Unit long axis, pointing down in the image.
    cv::Point2f top, bottom;
};
LightGeometry geometry(const cv::RotatedRect &rect) {
    const float angle = (rect.angle + (rect.size.width < rect.size.height ? 90.f : 0.f))
                        * static_cast<float>(CV_PI / 180.0);
    cv::Point2f axis(std::cos(angle), std::sin(angle));
    if (axis.y < 0) axis *= -1.f;
    const float length = std::max(rect.size.width, rect.size.height);
    return {length, std::min(rect.size.width, rect.size.height), axis,
            rect.center - axis * (length / 2), rect.center + axis * (length / 2)};
}
struct Candidate { size_t left, right; Armor armor; };
}

bool restoreLightEndpoints(Armor &armor, const std::vector<cv::RotatedRect> &refined,
                           const std::vector<cv::RotatedRect> &original)
{
    if (refined.size() != original.size()) return false;
    size_t left = refined.size(), right = refined.size();
    for (size_t i = 0; i < refined.size(); ++i) {
        if (cv::norm(refined[i].center-armor.left_light.center) < 1e-4) left = i;
        if (cv::norm(refined[i].center-armor.right_light.center) < 1e-4) right = i;
    }
    if (left == refined.size() || right == refined.size() || left == right) return false;
    const auto l = geometry(original[left]), r = geometry(original[right]);
    armor.left_light = original[left]; armor.right_light = original[right];
    armor.center = (original[left].center + original[right].center) / 2.f;
    armor.vertices[0] = l.top; armor.vertices[1] = l.bottom;
    armor.vertices[2] = r.bottom; armor.vertices[3] = r.top;
    armor.detection_score *= .9; // Mark a failed refinement as lower confidence.
    return true;
}

std::vector<Armor> matchArmors(const std::vector<cv::RotatedRect> &lightBars,
                              float max_angle_diff,
                              float max_length_ratio,
                              float min_aspect_ratio,
                              float max_y_diff_ratio,
                              float max_aspect_ratio,
                              float min_detection_score,
                              const std::vector<float> &light_quality)
{
    std::vector<Armor> armors;
    if (lightBars.size() < 2 || max_angle_diff <= 0 || max_length_ratio < 1 ||
        max_y_diff_ratio <= 0 || min_aspect_ratio <= 0 || max_aspect_ratio < min_aspect_ratio ||
        min_detection_score <= 0 || min_detection_score > 1)
        return armors;
    auto bars = lightBars;
    std::vector<size_t> order(bars.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (lightBars[a].center.x != lightBars[b].center.x)
            return lightBars[a].center.x < lightBars[b].center.x;
        return lightBars[a].center.y < lightBars[b].center.y;
    });
    std::vector<float> qualities;
    for (size_t i = 0; i < order.size(); ++i) {
        bars[i] = lightBars[order[i]];
        const float q = light_quality.size() == bars.size() ? light_quality[order[i]] : 1.f;
        qualities.push_back(std::isfinite(q) ? std::clamp(q, .1f, 1.f) : .1f);
    }
    std::vector<LightGeometry> lights;
    for (const auto &bar : bars) lights.push_back(geometry(bar));
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < bars.size(); ++i) {
        for (size_t j = i + 1; j < bars.size(); ++j) {
            const auto &l = lights[i];
            const auto &r = lights[j];
            if (l.length < 1 || r.length < 1 || l.width <= 0 || r.width <= 0) continue;
            const float avg_length = (l.length + r.length) / 2;
            const float angle_diff = std::acos(std::clamp(std::abs(l.axis.dot(r.axis)), 0.f, 1.f))
                                     * static_cast<float>(180.0 / CV_PI);
            const float length_ratio = std::max(l.length, r.length) / std::min(l.length, r.length);
            cv::Point2f vertical = l.axis + r.axis;
            const float axis_norm = static_cast<float>(cv::norm(vertical));
            if (axis_norm < 1e-6f) continue;
            vertical *= 1.f / axis_norm;
            const cv::Point2f horizontal(vertical.y, -vertical.x);
            const cv::Point2f delta = bars[j].center - bars[i].center;
            // Measure offsets in the plate's local axes, not the camera's axes.
            const float width = std::abs(delta.dot(horizontal)) / avg_length;
            const float y_offset = std::abs(delta.dot(vertical)) / avg_length;
            if (angle_diff > max_angle_diff || length_ratio > max_length_ratio ||
                y_offset > max_y_diff_ratio || width < min_aspect_ratio || width > max_aspect_ratio)
                continue;

            int intervening = 0;
            for (size_t k = i + 1; k < j; ++k) {
                const float offset = std::abs((bars[k].center - bars[i].center).dot(vertical));
                if (offset < avg_length * .5f && lights[k].length > avg_length * .5f)
                    ++intervening;
            }
            // Foreshortening is allowed; overly wide pairs and skewed endpoints cost more.
            const double wide_penalty = std::max(0.f, width - 2.5f) / .6;
            const double shape_penalty = .5 * (l.width / l.length + r.width / r.length);
            const double cost = .35 * std::pow(angle_diff / max_angle_diff, 2)
                              + .8 * std::pow(std::log(length_ratio), 2)
                              + 2.0 * std::pow(y_offset, 2)
                              + wide_penalty * wide_penalty + shape_penalty + 1.5 * intervening;
            const double score = std::exp(-cost) * std::sqrt(qualities[i]*qualities[j]);
            if (score < min_detection_score) continue;
            Armor armor;
            armor.left_light = bars[i]; armor.right_light = bars[j];
            armor.center = (bars[i].center + bars[j].center) / 2.f;
            armor.vertices[0] = l.top; armor.vertices[1] = l.bottom;
            armor.vertices[2] = r.bottom; armor.vertices[3] = r.top;
            armor.detection_score = score;
            candidates.push_back({i, j, armor});
        }
    }

    // Solve each independent conflict component. Unlike first-fit matching, a
    // locally early candidate cannot consume lights needed by a better pairing.
    std::vector<std::vector<size_t>> edges(bars.size());
    for (size_t e = 0; e < candidates.size(); ++e) {
        edges[candidates[e].left].push_back(e);
        edges[candidates[e].right].push_back(e);
    }
    std::vector<bool> visited(bars.size(), false);
    for (size_t root = 0; root < bars.size(); ++root) {
        if (visited[root] || edges[root].empty()) continue;
        std::vector<size_t> component{root}; visited[root] = true;
        for (size_t v = 0; v < component.size(); ++v) {
            for (size_t e : edges[component[v]]) {
                const auto &c = candidates[e];
                const size_t other = c.left == component[v] ? c.right : c.left;
                if (!visited[other]) { visited[other] = true; component.push_back(other); }
            }
        }
        if (component.size() <= 20) {
            std::unordered_map<size_t, size_t> local;
            for (size_t i = 0; i < component.size(); ++i) local[component[i]] = i;
            struct Choice { double score; int edge; };
            std::unordered_map<uint32_t, Choice> memo;
            std::function<double(uint32_t)> solve = [&](uint32_t mask) -> double {
                if (!mask) return 0.;
                if (memo.count(mask)) return memo.at(mask).score;
                size_t first = 0;
                while (!(mask & (1u << first))) ++first;
                const uint32_t remaining = mask & ~(1u << first);
                Choice best{solve(remaining), -1}; // A light may remain unmatched.
                for (size_t e : edges[component[first]]) {
                    const auto &c = candidates[e];
                    const size_t other = local.at(c.left == component[first] ? c.right : c.left);
                    if (!(remaining & (1u << other))) continue;
                    const double score = c.armor.detection_score + solve(remaining & ~(1u << other));
                    if (score > best.score) best = {score, static_cast<int>(e)};
                }
                memo[mask] = best;
                return best.score;
            };
            uint32_t mask = (1u << component.size()) - 1;
            solve(mask);
            while (mask) {
                size_t first = 0;
                while (!(mask & (1u << first))) ++first;
                const int edge = memo.at(mask).edge;
                mask &= ~(1u << first);
                if (edge >= 0) {
                    const auto &c = candidates[edge];
                    armors.push_back(c.armor);
                    mask &= ~(1u << local.at(c.left));
                    mask &= ~(1u << local.at(c.right));
                }
            }
        } else {
            // Bounded fallback for cluttered scenes: quality order, never x order.
            std::vector<size_t> component_edges;
            for (size_t v : component) for (size_t e : edges[v])
                if (candidates[e].left == v) component_edges.push_back(e);
            std::sort(component_edges.begin(), component_edges.end(), [&](size_t a, size_t b) {
                return candidates[a].armor.detection_score > candidates[b].armor.detection_score;
            });
            std::vector<bool> used(bars.size(), false);
            for (size_t e : component_edges) {
                const auto &c = candidates[e];
                if (!used[c.left] && !used[c.right]) {
                    armors.push_back(c.armor); used[c.left] = used[c.right] = true;
                }
            }
        }
    }
    std::sort(armors.begin(), armors.end(), [](const Armor &a, const Armor &b) {
        return a.center.x < b.center.x;
    });
    return armors;
}

cv::Mat drawArmors(const cv::Mat &src, const std::vector<Armor> &armors)
{
    cv::Mat out = src.clone();
    for (const auto &armor : armors)
    {
        for (int i = 0; i < 4; i++)
        {
            cv::line(out, armor.vertices[i], armor.vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }
    return out;
}
