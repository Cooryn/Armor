#include "predictor.hpp"
#include "predictor_polar.hpp"
#include "predictor_armor.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
static_assert(Eigen::Matrix<double, 6, 1>::RowsAtCompileTime == 6);
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
    ArmorEKF model{};
    Eigen::Matrix<double, 11, 1> s;
    s << .1, 0, .2, 0, 3, 0, theta, 0, .26, 0, 0;
    return armor_h(s, aid);
}
static ArmorEKF tracker() {
    ArmorEKF b{};
    b.initialize(measurement(0));
    return b;
}
static void single_plate_regressions() {
    SinglePlateEKF filter{};
    check(!filter.update({0, 0, 3}), "uninitialized single-plate update");
    bool caught = false;
    try {
        filter.predict(.01);
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    check(caught, "uninitialized single-plate prediction");

    for (const Eigen::Vector3d &position : std::array<Eigen::Vector3d, 5>{
             Eigen::Vector3d(.2, .1, 3), Eigen::Vector3d(-.4, -.6, 5), Eigen::Vector3d(.05, 3, .02),
             Eigen::Vector3d(1e-8, .2, -3), Eigen::Vector3d(-1e-8, -.2, -3)}) {
        Eigen::Matrix<double, 6, 1> truth;
        truth << position(0), .4, position(1), -.2, position(2), .3;
        Eigen::Matrix<double, 3, 6> numerical;
        for (int column = 0; column < 6; ++column) {
            Eigen::Matrix<double, 6, 1> plus = truth, minus = truth;
            plus(column) += 1e-6;
            minus(column) -= 1e-6;
            Eigen::Vector3d difference = single_plate_h(plus) - single_plate_h(minus);
            for (int angle = 0; angle < 2; ++angle)
                difference(angle) = single_plate_wrap_to_pi(difference(angle));
            numerical.col(column) = difference / 2e-6;
        }
        near(single_plate_jacobian(truth), numerical, 1e-7);
        filter.initialize(single_plate_h(truth));
        for (int axis = 0; axis < 3; ++axis) {
            check(std::abs(filter.state_(axis * 2) - position(axis)) < 1e-12, "spherical initialization");
            check(filter.state_(axis * 2 + 1) == 0, "zero initial velocity");
        }
        const auto state = filter.state_;
        check(filter.update(single_plate_h(state)), "zero innovation single-plate update");
        near(filter.state_, state, 1e-12);
    }

    // On the optical axis the range update reduces to a scalar Kalman result.
    filter.initialize({0, 0, 3});
    check(filter.update({0, 0, 3.1}), "single-plate range update");
    check(std::abs(filter.state_(4) - (3 + .1 * 10 / 10.16)) < 1e-12, "single-plate range gain");
    check(std::abs(filter.covariance_(4, 4) - 10 * .16 / 10.16) < 1e-12, "single-plate range variance");
    const auto state = filter.state_;
    const auto covariance = filter.covariance_;
    for (const Eigen::Vector3d &invalid : std::array<Eigen::Vector3d, 4>{
             Eigen::Vector3d(0, 0, -1), Eigen::Vector3d(0, single_plate_pi / 2, 3),
             Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0, 3)}) {
        check(!filter.update(invalid), "invalid single-plate observation");
        near(filter.state_, state, 0);
        near(filter.covariance_, covariance, 0);
        caught = false;
        try {
            filter.initialize(invalid);
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        check(caught, "invalid single-plate initialization");
        near(filter.state_, state, 0);
    }
    for (double dt : {-1., std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity(), 1e200}) {
        caught = false;
        try {
            filter.predict(dt);
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        check(caught, "invalid single-plate dt");
        near(filter.state_, state, 0);
        near(filter.covariance_, covariance, 0);
    }
    filter.predict(0);
    near(filter.state_, state, 0);
    near(filter.covariance_, covariance, 0);

    for (bool moving : {false, true}) {
        Eigen::Matrix<double, 6, 1> truth;
        truth << -.4, moving ? .3 : 0, .2, moving ? -.06 : 0, 3, moving ? .12 : 0;
        filter.initialize(single_plate_h(truth));
        for (int frame = 0; frame < 400; ++frame) {
            const double dt = std::array<double, 4>{.015, .033, .075, .2}[frame % 4];
            for (int axis = 0; axis < 3; ++axis)
                truth(axis * 2) += truth(axis * 2 + 1) * dt;
            filter.predict(dt);
            if (frame < 110 || frame >= 140)
                check(filter.update(single_plate_h(truth)), "single-plate trajectory update");
            near(filter.covariance_, filter.covariance_.transpose(), 1e-12);
            Eigen::LLT<Eigen::Matrix<double, 6, 6>> cholesky(filter.covariance_);
            check(cholesky.info() == Eigen::Success, "single-plate covariance positive definite");
        }
        near(filter.state_, truth, 1e-4);
    }
    filter.initialize({single_plate_pi - .001, .01, 3});
    check(filter.update({-single_plate_pi + .001, .01, 3}), "single-plate yaw wrap update");
    check(std::abs(filter.state_(0)) < .01, "single-plate yaw wrap uses short residual");

    // Predict a target onto the vertical axis: the undefined yaw must not corrupt state.
    filter.initialize({single_plate_pi / 2, 0, 1});
    filter.predict(.1);
    check(filter.update({single_plate_pi / 2, 0, .9}), "single-plate approach to origin");
    filter.predict(-filter.state_(0) / filter.state_(1));
    const auto singular_state = filter.state_;
    const auto singular_covariance = filter.covariance_;
    check(!filter.update({single_plate_pi / 2, 0, .9}), "singular single-plate prior skips update");
    near(filter.state_, singular_state, 0);
    near(filter.covariance_, singular_covariance, 0);
}
static void eigen_filter_regressions() {
    // Compare the bounded Eigen joint matrices with an independent dynamic LU solve.
    for (int count = 1; count <= 4; ++count) {
        auto filter = tracker();
        filter.predict(.037);
        std::vector<Observation> observations;
        for (int id = 0; id < count; ++id) {
            auto z = measurement(.02, id);
            z(2) += .01 * (id + 1);
            observations.push_back({z, id});
        }
        const auto matches = filter.associate(observations).first;
        check(matches.size() == static_cast<size_t>(count), "one to four joint plates");
        const int n = count * 4;
        Eigen::MatrixXd H(n, 11), residual(n, 1), R = Eigen::MatrixXd::Zero(n, n);
        for (int k = 0; k < count; ++k) {
            H.middleRows(4 * k, 4) = matches[k].H;
            residual.middleRows(4 * k, 4) = matches[k].residual;
            R.block(4 * k, 4 * k, 4, 4) = filter.R;
            const Eigen::Matrix4d S = matches[k].H * filter.P * matches[k].H.transpose() + filter.R;
            check(std::abs(matches[k].nis - matches[k].residual.dot(S.partialPivLu().solve(matches[k].residual))) < 1e-10,
                  "LDLT innovation agrees with LU");
        }
        const Eigen::MatrixXd S = H * filter.P * H.transpose() + R;
        const Eigen::MatrixXd K = S.partialPivLu().solve(H * filter.P).transpose();
        const Eigen::Matrix<double, 11, 1> expected_state = filter.X + K * residual;
        const Eigen::Matrix<double, 11, 11> A = Eigen::Matrix<double, 11, 11>::Identity() - K * H;
        const Eigen::Matrix<double, 11, 11> expected_covariance = A * filter.P * A.transpose() + K * R * K.transpose();
        check(filter.update_multi(observations).first.size() == static_cast<size_t>(count), "joint update count");
        near(filter.X, expected_state);
        near(filter.P, expected_covariance);
        near(filter.P, filter.P.transpose(), 1e-12);
        check(Eigen::LLT<Eigen::Matrix<double, 11, 11>>(filter.P).info() == Eigen::Success,
              "joint covariance positive definite");
    }

    PolarEKF polar{};
    polar.X << .1, .2, .3, -.1, 3., .4, polar_pi - .01, 1., .26;
    polar.predict(.023);
    auto z = polar_h(polar.X, 1);
    z(0) += .01;
    z(2) += .03;
    const auto H = polar_get_jacobian(polar.X, 1);
    const auto residual = polar_angular_residual(z, polar_h(polar.X, 1));
    const Eigen::Matrix4d S = H * polar.P * H.transpose() + polar.R;
    const Eigen::Matrix<double, 9, 4> K = (polar.P * H.transpose()) * S.partialPivLu().solve(Eigen::Matrix4d::Identity());
    Eigen::Matrix<double, 9, 1> expected_state = polar.X + K * residual;
    expected_state(6) = polar_wrap_to_pi(expected_state(6));
    const Eigen::Matrix<double, 9, 9> expected_covariance = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * polar.P;
    polar.update(z);
    near(polar.X, expected_state);
    near(polar.P, expected_covariance);

    // A singular covariance is an error; no alternate solve or noise inflation is used.
    auto armor = tracker();
    armor.P.setZero();
    armor.R.setZero();
    const auto armor_state = armor.X;
    bool caught = false;
    try { armor.update_multi({Observation{measurement(0)}}); }
    catch (const std::runtime_error &) { caught = true; }
    check(caught, "singular armor covariance rejected");
    near(armor.X, armor_state, 0);
    near(armor.P, Eigen::Matrix<double, 11, 11>::Zero(), 0);
    polar.P.setZero();
    polar.R.setZero();
    const auto polar_state = polar.X;
    caught = false;
    try { polar.update(polar_h(polar.X, 1)); }
    catch (const std::runtime_error &) { caught = true; }
    check(caught, "singular polar covariance rejected");
    near(polar.X, polar_state, 0);
    near(polar.P, Eigen::Matrix<double, 9, 9>::Zero(), 0);
}
int main() {
    try {
        single_plate_regressions();
        eigen_filter_regressions();
        check(armor_wrap_to_pi(armor_pi) == -armor_pi, "armor_pi boundary");
        check(polar_round_even(.5) == 0 && polar_round_even(1.5) == 2 && polar_round_even(-.5) == 0 &&
                  polar_round_even(-1.5) == -2,
              "Python rounding");
        auto b = tracker();
        ArmorEKF base_tracker{};
        base_tracker.base_frame = true;
        const Eigen::Vector4d rear_observation(2.2, .1, 3., .3);
        base_tracker.initialize(rear_observation);
        near(armor_h(base_tracker.X, 0), rear_observation);
        near(armor_h(b.X, 0), measurement(0));
        b.predict(.1);
        auto x = b.X;
        auto p = b.P;
        auto outlier = measurement(0);
        outlier(2) += 2;
        check(b.update_multi({Observation{outlier}}).first.empty(), "range gate");
        near(b.X, x, 0);
        near(b.P, p, 0);
        check(b.update_multi({Observation{Eigen::Vector4d::Zero()}}).first.empty(), "invalid distance");
        check(b.update_multi({Observation{Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN())}}).first.empty(), "invalid NaN");
        check(b.update_multi({Observation{measurement(0), 9}}).first.empty(), "invalid id");
        check(b.associate({Observation{measurement(0), -2}}).first.empty(),
              "only -1 is an automatic plate id");
        const auto id_reference = tracker();
        const auto automatic = id_reference.associate({Observation{measurement(0, 1), -1}}).first;
        check(automatic.size() == 1 && automatic[0].armor_id == 1, "automatic plate id sentinel");
        check(id_reference.associate({Observation{measurement(0, 1), 0}}).first.empty(),
              "specified plate zero must not become automatic");
        auto a = tracker();
        b = tracker();
        std::vector<Observation> observations = {{measurement(.02), 0}, {measurement(.02, 1), 1}};
        auto matches = a.associate(observations).first;
        check(matches.size() == 2, "stacked reference association");
        Eigen::MatrixXd stacked_h(8, 11), stacked_residual(8, 1), stacked_r = Eigen::MatrixXd::Zero(8, 8);
        for (int k = 0; k < 2; ++k) {
            stacked_h.middleRows(4*k, 4) = matches[k].H;
            stacked_residual.middleRows(4*k, 4) = matches[k].residual;
            stacked_r.block(4*k, 4*k, 4, 4) = a.R;
        }
        Eigen::MatrixXd gain = (stacked_h*a.P*stacked_h.transpose() + stacked_r).ldlt().solve(stacked_h*a.P).transpose();
        Eigen::MatrixXd expected_state = a.X + gain*stacked_residual;
        Eigen::MatrixXd identity_minus_kh = Eigen::MatrixXd::Identity(11, 11) - gain*stacked_h;
        Eigen::MatrixXd expected_covariance = identity_minus_kh*a.P*identity_minus_kh.transpose() + gain*stacked_r*gain.transpose();
        check(a.update_multi(observations).first.size() == 2, "two plate update");
        near(a.X, expected_state);
        near(a.P, expected_covariance);
        std::reverse(observations.begin(), observations.end());
        b.update_multi(observations);
        near(a.X, b.X);
        near(a.P, b.P);
        b = tracker();
        auto result = b.update_multi({Observation{measurement(0)}, Observation{measurement(0)},
                                      Observation{measurement(0, 1)}});
        check(result.first.size() == 2, "unique ids");
        b = tracker();
        b.P = b.P * 100;
        auto bad = measurement(0, 1);
        bad(3) = 0;
        check(b.associate({Observation{measurement(0), 0}, Observation{bad, 1}}).first.size() == 1,
              "pair geometry");
        b = tracker();
        b.predict(.03);
        bad = measurement(0);
        bad(2) += .1;
        result = b.associate({Observation{bad, 0}, Observation{measurement(0), 0}});
        check(result.first.size() == 1 && result.first[0].index == 1, "lowest NIS conflict selection");
        b = tracker();
        b.X(1) = 1;
        b.X(3) = -.2;
        b.X(5) = .3;
        b.X(7) = 8;
        b.X(6) = armor_pi - .1;
        b.X(9) = .02;
        b.X(10) = .04;
        x = b.X;
        p = b.P;
        auto f = b.F;
        auto forecast = b.forecast(.05);
        check(std::abs(forecast.state(0) - (x(0) + .05)) < 1e-12 &&
                  std::abs(forecast.state(2) - (x(2) - .01)) < 1e-12 &&
                  std::abs(forecast.state(4) - (x(4) + .015)) < 1e-12 &&
                  std::abs(forecast.state(6) - (-armor_pi + .3)) < 1e-12,
              "forecast translation and yaw wrap");
        near(b.X, x, 0);
        near(b.P, p, 0);
        near(b.F, f, 0);
        a = b;
        a.predict(.05);
        near(a.X, forecast.state);
        near(a.P, forecast.covariance);
        near(b.forecast(0).state, b.X, 1e-12);
        for (int id = 0; id < 4; ++id) {
            double yaw = forecast.state(6) + id * armor_pi / 2;
            check(std::abs(
                      forecast.plates(id, 0) -
                      (forecast.state(0) + (forecast.state(8) + (id % 2 ? forecast.state(9) : 0)) *
                                               std::sin(yaw))) < 1e-12,
                  "forecast plate");
            check(std::abs(forecast.plates(id, 1) -
                           (forecast.state(2) + (id % 2 ? .04 : 0))) < 1e-12 &&
                      std::abs(forecast.plates(id, 2) -
                           (forecast.state(4) - (.26 + (id % 2 ? .02 : 0)) * std::cos(yaw))) < 1e-12,
                  "forecast alternating heights and radii");
        }
        for (double dt : {-1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            bool caught = false;
            try {
                b.forecast(dt);
            } catch (const std::invalid_argument &) {
                caught = true;
            }
            check(caught, "invalid horizon");
        }
        bool caught = false;
        try {
            (ArmorEKF{}).forecast();
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        check(caught, "uninitialized forecast");
        b = tracker();
        b.X(7) = 2;
        for (int frame = 1; frame <= 200; ++frame) {
            b.predict(.02);
            double theta = 2 * frame * .02;
            if (frame < 60 || frame >= 80) {
                int aid = 0;
                for (int i = 1; i < 4; ++i)
                    if (std::abs(armor_wrap_to_pi(theta + i * armor_pi / 2)) <
                        std::abs(armor_wrap_to_pi(theta + aid * armor_pi / 2)))
                        aid = i;
                const auto associated = b.update_multi({Observation{measurement(theta, aid)}}).first;
                check(associated.size() == 1 && associated[0].armor_id == aid, "rotation id");
            } else
                b.update_multi({});
            near(b.P, b.P.transpose(), 1e-12); // Cholesky verifies positive definite covariance.
            Eigen::LLT<Eigen::Matrix<double, 11, 11>> cholesky(b.P);
            check(cholesky.info() == Eigen::Success, "covariance positive definite");
        }
        check(std::abs(b.X(0) - .1) < 1e-5 && std::abs(b.X(2) - .2) < 1e-5 &&
                  std::abs(b.X(4) - 3) < 1e-5 && std::abs(b.X(7) - 2) < 1e-5,
              "rotation convergence");
        PolarEKF polar{};
        polar.X << .1, .2, .3, -.1, 3., .4, armor_pi - .01, 1., .26;
        polar.predict(.02);
        check(std::abs(polar.X(0) - .104) < 1e-12 &&
                  std::abs(polar.X(2) - .298) < 1e-12 &&
                  std::abs(polar.X(4) - 3.008) < 1e-12 &&
                  std::abs(polar.X(6) - (-armor_pi + .01)) < 1e-12,
              "polar constant velocity and yaw wrap");
        const Eigen::Matrix<double, 9, 1> polar_state = polar.X;
        polar.update(polar_h(polar.X, 1));
        near(polar.X, polar_state); // Exact predicted observation has zero innovation.
        std::cout << "All C++ API regression checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
