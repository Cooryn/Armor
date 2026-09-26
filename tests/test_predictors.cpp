#include "predictor.hpp"
#include "predictor_polar.hpp"
#include "predictor_armor.hpp"
#include <iostream>
using namespace predictor;
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
static Eigen::MatrixXd measurement(double theta, int aid = 0) {
    ArmorEKF model;
    Eigen::MatrixXd s = ArmorEKF::make_vector({.1, 0, .2, 0, 3, 0, theta, 0, .26, 0, 0});
    return model.h(s, aid);
}
static ArmorEKF tracker() {
    ArmorEKF b;
    b.initialize(measurement(0));
    return b;
}
int main() {
    try {
        Eigen::MatrixXd system(2, 2), rhs(2, 2), expected(2, 2);
        system << 0, 2, 1, 3; // Requires pivoting, with two right-hand sides.
        expected << 1, -2, 3, 4;
        rhs = system * expected;
        near(ArmorEKF::solve(system, rhs), expected);
        check(std::abs(ArmorEKF::logabsdet(system) - std::log(2.)) < 1e-12, "Eigen log determinant");
        bool singular_rejected = false;
        try {
            ArmorEKF::solve(Eigen::MatrixXd::Zero(2, 2), rhs);
        } catch (const std::runtime_error &) {
            singular_rejected = true;
        }
        check(singular_rejected, "singular solve must throw");
        check(ArmorEKF::wrap_to_pi(ArmorEKF::pi) == -ArmorEKF::pi, "ArmorEKF::pi boundary");
        check(PolarEKF::round_even(.5) == 0 && PolarEKF::round_even(1.5) == 2 && PolarEKF::round_even(-.5) == 0 &&
                  PolarEKF::round_even(-1.5) == -2,
              "Python rounding");
        auto b = tracker();
        ArmorEKF base_tracker;
        base_tracker.base_frame = true;
        auto rear_observation = ArmorEKF::make_vector({2.2, .1, 3., .3});
        base_tracker.initialize(rear_observation);
        near(base_tracker.h(base_tracker.X, 0), rear_observation);
        near(b.h(b.X, 0), measurement(0));
        b.predict(.1);
        auto x = b.X;
        auto p = b.P;
        auto outlier = measurement(0);
        outlier(2) += 2;
        check(!b.update(outlier), "range gate");
        near(b.X, x, 0);
        near(b.P, p, 0);
        check(!b.update(Eigen::MatrixXd::Zero(3, 1)), "invalid shape");
        check(!b.update(Eigen::MatrixXd::Zero(4, 1)), "invalid distance");
        check(!b.update(Eigen::MatrixXd::Constant(4, 1, std::numeric_limits<double>::quiet_NaN())), "invalid NaN");
        check(!b.update(measurement(0), 9), "invalid id");
        auto a = tracker();
        b = tracker();
        std::vector<Observation> observations = {{measurement(.02), 0, .4, 1.5},
                                                 {measurement(.02, 1), 1, .95, .1}};
        check(a.update_multi(observations).first.size() == 2, "two plate update");
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
        result = b.associate(
            {Observation{measurement(0), 0, .1, 3}, Observation{measurement(0), 0, 1, 0}});
        check(result.first.size() == 1 && result.first[0].index == 1, "noise volume cost");
        auto noise = b.observation_noise({measurement(.5), {}, 0, 1e6});
        for (int i = 0; i < 4; ++i)
            check(noise.second(i) >= 1 && noise.second(i) <= (i == 3   ? 6
                                                              : i == 2 ? 4.5
                                                                       : 3),
                  "noise bounds");
        b = tracker();
        b.X(1) = 1;
        b.X(3) = -.2;
        b.X(5) = .3;
        b.X(7) = 8;
        b.X(6) = ArmorEKF::pi - .1;
        b.X(9) = .02;
        b.X(10) = .04;
        x = b.X;
        p = b.P;
        auto f = b.F;
        auto forecast = b.forecast(.05);
        check(std::abs(forecast.state(0) - (x(0) + .05)) < 1e-12 &&
                  std::abs(forecast.state(2) - (x(2) - .01)) < 1e-12 &&
                  std::abs(forecast.state(4) - (x(4) + .015)) < 1e-12 &&
                  std::abs(forecast.state(6) - (-ArmorEKF::pi + .3)) < 1e-12,
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
            double yaw = forecast.state(6) + id * ArmorEKF::pi / 2;
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
            ArmorEKF().forecast();
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
                    if (std::abs(ArmorEKF::wrap_to_pi(theta + i * ArmorEKF::pi / 2)) <
                        std::abs(ArmorEKF::wrap_to_pi(theta + aid * ArmorEKF::pi / 2)))
                        aid = i;
                check(b.update(measurement(theta, aid)) == aid, "rotation id");
            } else
                b.update_multi({});
            near(b.P, b.P.transpose(), 1e-12); // Cholesky verifies positive definite covariance.
            Eigen::LLT<Eigen::MatrixXd> cholesky(b.P);
            check(cholesky.info() == Eigen::Success, "covariance positive definite");
        }
        check(std::abs(b.X(0) - .1) < 1e-5 && std::abs(b.X(2) - .2) < 1e-5 &&
                  std::abs(b.X(4) - 3) < 1e-5 && std::abs(b.X(7) - 2) < 1e-5,
              "rotation convergence");
        BasicPredictor basic;
        auto state = basic.state;
        auto covariance = basic.P;
        basic.predict(.03);
        near(state, basic.state, 0);
        near(covariance, basic.P, 0);
        // Independent scalar Kalman result: diagonal P and direct position observation.
        basic.update(ArmorEKF::make_vector({1, 2, 3}), state, covariance);
        for (int axis = 0; axis < 3; ++axis) {
            check(std::abs(basic.state(2 * axis) - (axis + 1) * 10. / 10.1) < 1e-12,
                  "basic Kalman gain");
            check(std::abs(basic.P(2 * axis, 2 * axis) - 10. / 101.) < 1e-12,
                  "basic posterior covariance");
        }
        PolarEKF polar;
        polar.X = ArmorEKF::make_vector({.1, .2, .3, -.1, 3., .4, ArmorEKF::pi - .01, 1., .26});
        polar.predict(.02);
        check(std::abs(polar.X(0) - .104) < 1e-12 &&
                  std::abs(polar.X(2) - .298) < 1e-12 &&
                  std::abs(polar.X(4) - 3.008) < 1e-12 &&
                  std::abs(polar.X(6) - (-ArmorEKF::pi + .01)) < 1e-12,
              "polar constant velocity and yaw wrap");
        x = polar.X;
        polar.update(polar.h(polar.X, 1));
        near(polar.X, x); // Exact predicted observation has zero innovation.
        std::cout << "All C++ API regression checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
