#include "predictor_single.hpp"
#include "predictor_armor.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
static_assert(Eigen::Matrix<double, 8, 1>::RowsAtCompileTime == 8);
static_assert(Eigen::Matrix<double, 9, 1>::RowsAtCompileTime == 9);
static_assert(Eigen::Matrix<double, 11, 1>::RowsAtCompileTime == 11);
static_assert(Eigen::Matrix<double, 4, 11>::RowsAtCompileTime == 4 && Eigen::Matrix<double, 4, 11>::ColsAtCompileTime == 11);
static_assert(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 16, 16>::MaxRowsAtCompileTime == 16);
static void check(bool v, const char *msg) {
    if (!v)
        throw std::runtime_error(msg);
}
static void near(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, double tolerance = 1e-10) {
    check(a.rows() == b.rows() && a.cols() == b.cols(), "shape");
    for (Eigen::Index i = 0; i < a.size(); ++i)
        if (!std::isfinite(a.data()[i]) || !std::isfinite(b.data()[i]) ||
            std::abs(a.data()[i] - b.data()[i]) > tolerance)
            throw std::runtime_error("matrix mismatch at " + std::to_string(i) +
                                     " difference=" + std::to_string(a.data()[i] - b.data()[i]) +
                                     " tolerance=" + std::to_string(tolerance));
}
static Eigen::Vector4d measurement(double theta, int aid = 0) {
    ::predictor_armor model{};
    Eigen::Matrix<double, 11, 1> s;
    s << .1, 0, .2, 0, 3, 0, theta, 0, .26, 0, 0;
    return predictor_armor::h(s, aid);
}
static ::predictor_armor tracker() {
    ::predictor_armor b{};
    b.initialize(measurement(0));
    return b;
}
static void single_plate_regressions() {
    ::predictor_single filter{};
    for (const Eigen::Vector3d &position : std::array<Eigen::Vector3d, 5>{
             Eigen::Vector3d(.2, .1, 3), Eigen::Vector3d(-.4, -.6, 5), Eigen::Vector3d(.05, 3, .02),
             Eigen::Vector3d(1e-8, .2, -3), Eigen::Vector3d(-1e-8, -.2, -3)}) {
        Eigen::Matrix<double, 8, 1> truth;
        truth << position(0), .4, position(1), -.2, position(2), .3, .4, 0;
        Eigen::Matrix<double, 4, 8> numerical;
        for (int column = 0; column < 8; ++column) {
            Eigen::Matrix<double, 8, 1> plus = truth, minus = truth;
            plus(column) += 1e-6;
            minus(column) -= 1e-6;
            Eigen::Vector4d difference = predictor_single::h(plus) - predictor_single::h(minus);
            for (int angle : {0, 1, 3})
                difference(angle) = predictor_single::wrap_to_pi(difference(angle));
            numerical.col(column) = difference / 2e-6;
        }
        near(predictor_single::jacobian(truth), numerical, 1e-7);
        filter.initialize(predictor_single::h(truth));
        for (int axis = 0; axis < 3; ++axis) {
            check(std::abs(filter.X(axis * 2) - position(axis)) < 1e-12, "spherical initialization");
            check(filter.X(axis * 2 + 1) == 0, "zero initial velocity");
        }
        const auto state = filter.X;
        check(filter.update(predictor_single::h(state)), "zero innovation single-plate update");
        near(filter.X, state, 1e-12);
    }

    filter.initialize({0, 0, 3, 0});
    check(filter.update({0, 0, 3.1, 0}), "single-plate range update");
    const double range_noise = predictor_single{}.R(2, 2);
    check(std::abs(filter.X(4) - (3 + .1 * 10 / (10 + range_noise))) < 1e-12, "single-plate range gain");
    check(std::abs(filter.P(4, 4) - 10 * range_noise / (10 + range_noise)) < 1e-12, "single-plate range variance");
    const auto state = filter.X;
    const auto covariance = filter.P;
    filter.predict(0);
    near(filter.X, state, 0);
    near(filter.P, covariance, 0);

    for (bool moving : {false, true}) {
        Eigen::Matrix<double, 8, 1> truth;
        truth << -.4, moving ? .3 : 0, .2, moving ? -.06 : 0, 3, moving ? .12 : 0, .4, 0;
        filter.initialize(predictor_single::h(truth));
        for (int frame = 0; frame < 400; ++frame) {
            const double dt = std::array<double, 4>{.015, .033, .075, .2}[frame % 4];
            for (int axis = 0; axis < 3; ++axis)
                truth(axis * 2) += truth(axis * 2 + 1) * dt;
            filter.predict(dt);
            if (frame < 110 || frame >= 140)
                check(filter.update(predictor_single::h(truth)), "single-plate trajectory update");
            near(filter.P, filter.P.transpose(), 1e-12);
            Eigen::LLT<Eigen::Matrix<double, 8, 8>> cholesky(filter.P);
            check(cholesky.info() == Eigen::Success, "single-plate covariance positive definite");
        }
        near(filter.X, truth, 1e-4);
    }
    filter.initialize({predictor_single::pi - .001, .01, 3, 0});
    check(filter.update({-predictor_single::pi + .001, .01, 3, 0}), "single-plate yaw wrap update");
    check(std::abs(filter.X(0)) < .01, "single-plate yaw wrap uses short residual");

    filter.initialize({0, 0, 3, predictor_single::pi - .01});
    filter.X(7) = .6;
    filter.predict(.05);
    check(filter.update({0, 0, 3, -predictor_single::pi + .02}), "armor yaw boundary update");
    check(std::abs(filter.X(7) - .6) < 1e-10, "yaw boundary preserves angular velocity");
    for (int frame = 0; frame < 200; ++frame) {
        filter.predict(.03);
        const double yaw = predictor_single::wrap_to_pi(-predictor_single::pi + .02 + .6 * (frame + 1) * .03);
        check(filter.update({0, 0, 3, yaw}), "constant angular velocity update");
    }
    check(std::abs(filter.X(7) - .6) < 1e-6, "angular velocity convergence");
    predictor_single rotating;
    rotating.update_frame(0, {{0, 0, 3, .2}}, 50);
    rotating.X(7) = .6;
    const auto missed = rotating.update_frame(100, {}, 50);
    check(std::abs(missed.current.plates[0](3) - .26) < 1e-12 &&
          std::abs(missed.future.plates[0](3) - .29) < 1e-12,
          "missed frame advances current and future yaw");
    check(std::abs(rotating.X(6) - .26) < 1e-12, "future yaw does not mutate state");

    filter.initialize({predictor_single::pi / 2, 0, 1, 0});
    filter.predict(.1);
    check(filter.update({predictor_single::pi / 2, 0, .9, 0}), "single-plate approach to origin");
    filter.predict(-filter.X(0) / filter.X(1));
    const auto singular_state = filter.X;
    const auto singular_covariance = filter.P;
    check(!filter.update({predictor_single::pi / 2, 0, .9, 0}), "singular single-plate prior skips update");
    near(filter.X, singular_state, 0);
    near(filter.P, singular_covariance, 0);
}
static void eigen_filter_regressions() {
    for (int count = 1; count <= 4; ++count) {
        auto filter = tracker();
        filter.predict(.037);
        std::vector<Eigen::Vector4d> observations;
        for (int id = 0; id < count; ++id) {
            auto z = measurement(.02, id);
            z(2) += .01 * (id + 1);
            observations.push_back(z);
        }
        const auto matches = filter.associate(observations);
        check(matches.size() == static_cast<size_t>(count), "one to four joint plates");
        const int n = count * 4;
        Eigen::MatrixXd H(n, 11), residual(n, 1), R = Eigen::MatrixXd::Zero(n, n);
        for (int k = 0; k < count; ++k) {
            H.middleRows(4 * k, 4) = predictor_armor::jacobian(filter.X, matches[k].armor_id);
            residual.middleRows(4 * k, 4) = matches[k].residual;
            R.block(4 * k, 4 * k, 4, 4) = filter.R;

        }
        const Eigen::MatrixXd S = H * filter.P * H.transpose() + R;
        const Eigen::MatrixXd K = S.partialPivLu().solve(H * filter.P).transpose();
        const Eigen::Matrix<double, 11, 1> expected_state = filter.X + K * residual;
        const Eigen::Matrix<double, 11, 11> A = Eigen::Matrix<double, 11, 11>::Identity() - K * H;
        const Eigen::Matrix<double, 11, 11> expected_covariance = A * filter.P * A.transpose() + K * R * K.transpose();
        check(filter.update(observations).size() == static_cast<size_t>(count), "joint update count");
        near(filter.X, expected_state);
        near(filter.P, expected_covariance);
        near(filter.P, filter.P.transpose(), 1e-12);
        check(Eigen::LLT<Eigen::Matrix<double, 11, 11>>(filter.P).info() == Eigen::Success,
              "joint covariance positive definite");
    }


}
static void single_plate_response() {
    // One physical plate: sustained acceleration/reversal at camera frame rates.
    // These bounds reject a smooth but lagging filter, and limit static jitter.
    for (const int fps : {30, 60}) for (const bool moving : {false, true}) {
        predictor_single filter;
        std::mt19937 random(20261010 + fps + moving);
        std::normal_distribution<double> noise(0, 1);
        const auto position = [moving](double t) -> Eigen::Vector3d {
            if (!moving) return {.2, .1, 3};
            const double pi = predictor_single::pi;
            return {.35 * std::sin(2*pi*.65*t) + .12 * std::sin(2*pi*1.1*t),
                    .1 + .08 * std::sin(2*pi*.4*t), 3 + .2 * std::sin(2*pi*.35*t)};
        };
        double current_error = 0, future_error = 0;
        int count = 0;
        for (int frame = 0; frame < fps * 12; ++frame) {
            const double t = double(frame) / fps;
            Eigen::Matrix<double, 8, 1> truth = Eigen::Matrix<double, 8, 1>::Zero();
            const auto p = position(t);
            truth(0) = p(0); truth(2) = p(1); truth(4) = p(2);
            Eigen::Vector4d z;
            z = predictor_single::h(truth);
            z(0) += .003 * noise(random); z(1) += .001 * noise(random); z(2) += .035 * noise(random);
            z(3) = 0;
            const auto result = filter.update_frame(t * 1000, {z}, 50);
            if (t >= 1) {
                current_error += (result.current.plates[0].head<3>() - p).squaredNorm();
                future_error += (result.future.plates[0].head<3>() - position(t + .05)).squaredNorm();
                ++count;
            }
        }
        check(std::sqrt(current_error / count) < .035, "single plate motion response/static jitter");
        check(std::sqrt(future_error / count) < .06, "single plate 50ms forecast response");
    }
}

int main() {
    try {
        single_plate_regressions();
        single_plate_response();
        eigen_filter_regressions();
        check(predictor_armor::wrap_to_pi(predictor_armor::pi) == -predictor_armor::pi, "predictor_armor::pi boundary");
        auto b = tracker();
        ::predictor_armor base_tracker{};
        const Eigen::Vector4d rear_observation(2.2, .1, 3., .3);
        base_tracker.initialize(rear_observation);
        near(predictor_armor::h(base_tracker.X, 0), rear_observation);
        near(predictor_armor::h(b.X, 0), measurement(0));
        b.predict(.1);
        auto x = b.X;
        auto p = b.P;
        auto outlier = measurement(0);
        outlier(2) += 2;
        check(b.update({outlier}).empty(), "range gate");
        near(b.X, x, 0);
        near(b.P, p, 0);
        const auto automatic = b.associate({measurement(0, 1)});
        check(automatic.size() == 1 && automatic[0].armor_id == 1, "automatic plate id");
        auto a = tracker();
        b = tracker();
        std::vector<Eigen::Vector4d> observations = {measurement(.02), measurement(.02, 1)};
        auto matches = a.associate(observations);
        check(matches.size() == 2, "stacked reference association");
        Eigen::MatrixXd stacked_h(8, 11), stacked_residual(8, 1), stacked_r = Eigen::MatrixXd::Zero(8, 8);
        for (int k = 0; k < 2; ++k) {
            stacked_h.middleRows(4*k, 4) = predictor_armor::jacobian(a.X, matches[k].armor_id);
            stacked_residual.middleRows(4*k, 4) = matches[k].residual;
            stacked_r.block(4*k, 4*k, 4, 4) = a.R;
        }
        Eigen::MatrixXd gain = (stacked_h*a.P*stacked_h.transpose() + stacked_r).ldlt().solve(stacked_h*a.P).transpose();
        Eigen::MatrixXd expected_state = a.X + gain*stacked_residual;
        Eigen::MatrixXd identity_minus_kh = Eigen::MatrixXd::Identity(11, 11) - gain*stacked_h;
        Eigen::MatrixXd expected_covariance = identity_minus_kh*a.P*identity_minus_kh.transpose() + gain*stacked_r*gain.transpose();
        check(a.update(observations).size() == 2, "two plate update");
        near(a.X, expected_state);
        near(a.P, expected_covariance);
        std::reverse(observations.begin(), observations.end());
        b.update(observations);
        near(a.X, b.X);
        near(a.P, b.P);
        b = tracker();
        auto result = b.update({measurement(0), measurement(0),
                                      measurement(0, 1)});
        check(result.size() == 2, "unique ids");
        b = tracker();
        b.P = b.P * 100;
        auto bad = measurement(0, 1);
        bad(3) = 0;
        check(b.associate({measurement(0), bad}).size() == 1,
              "duplicate plate rejected");
        auto first = measurement(0), second = measurement(0, 1);
        first(3) += .3;
        second(3) -= .3;
        const auto independent_yaws = b.associate({first, second});
        check(independent_yaws.size() == 2 && independent_yaws[0].armor_id != independent_yaws[1].armor_id,
              "individually valid yaw observations can update together");
        b = tracker();
        b.predict(.03);
        bad = measurement(0);
        bad(2) += .1;
        result = b.associate({bad, measurement(0)});
        check(result.size() == 1 && result[0].index == 1, "nearest conflict selection");
        b = tracker();
        auto lateral = measurement(0);
        lateral(0) += .3;
        check(b.associate({lateral}).empty(), "lateral distance gate");
        b.max_yaw_error = .2;
        auto rotated = measurement(0);
        rotated(3) += b.max_yaw_error + .01;
        check(b.associate({rotated}).empty(), "absolute yaw gate");
        const auto tied = b.associate({measurement(0), measurement(0)});
        check(tied.size() == 1 && tied[0].index == 0, "equal distance keeps first observation");
        auto uncertain = b;
        uncertain.P *= 1000;
        const auto covariance_independent = uncertain.associate({bad, measurement(0)});
        check(covariance_independent.size() == 1 && covariance_independent[0].index == 1,
              "association does not depend on covariance");
        b = tracker();
        b.X(1) = 1;
        b.X(3) = -.2;
        b.X(5) = .3;
        b.X(7) = 8;
        b.X(6) = predictor_armor::pi - .1;
        b.X(9) = .02;
        b.X(10) = .04;
        x = b.X;
        p = b.P;
        a = b;
        a.update_frame(-1, {}, 0);
        const auto prediction = b.update_frame(-1, {}, 50);
        near(b.X, a.X, 0);
        near(b.P, a.P, 0);
        a = b;
        a.predict(.05);
        near(*prediction.future.center, Eigen::Vector3d(x(0) + .05, x(2) - .01, x(4) + .015));
        check(std::abs(prediction.future.plates[0](3) - (-predictor_armor::pi + .3)) < 1e-12,
              "future yaw wrap");
        const auto zero = b.update_frame(-1, {}, 0);
        for (int id = 0; id < 4; ++id) {
            near(prediction.future.plates[id], predictor_armor::plate_pose(a.X, id));
            near(zero.future.plates[id], predictor_armor::plate_pose(b.X, id));
        }
        b = tracker();
        b.X(7) = 2;
        for (int frame = 1; frame <= 200; ++frame) {
            b.predict(.02);
            double theta = 2 * frame * .02;
            if (frame < 60 || frame >= 80) {
                int aid = 0;
                for (int i = 1; i < 4; ++i)
                    if (std::abs(predictor_armor::wrap_to_pi(theta + i * predictor_armor::pi / 2)) <
                        std::abs(predictor_armor::wrap_to_pi(theta + aid * predictor_armor::pi / 2)))
                        aid = i;
                const auto associated = b.update({measurement(theta, aid)});
                check(associated.size() == 1 && associated[0].armor_id == aid, "rotation id");
            } else
                b.update({});
            near(b.P, b.P.transpose(), 1e-12);
            Eigen::LLT<Eigen::Matrix<double, 11, 11>> cholesky(b.P);
            check(cholesky.info() == Eigen::Success, "covariance positive definite");
        }
        check(std::abs(b.X(0) - .1) < 1e-5 && std::abs(b.X(2) - .2) < 1e-5 &&
                  std::abs(b.X(4) - 3) < 1e-5 && std::abs(b.X(7) - 2) < 1e-5,
              "rotation convergence");
        std::cout << "All C++ API regression checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
