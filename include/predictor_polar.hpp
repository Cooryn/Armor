#pragma once
#include "output.hpp"


class predictor_polar
{
public:
    static constexpr double pi = 3.14159265358979323846;
    static double wrap_to_pi(double angle);
    static Eigen::Vector4d angular_residual(const Eigen::Vector4d &observed, const Eigen::Vector4d &predicted);
    static int round_even(double x);
    static Eigen::Vector4d h(const Eigen::Matrix<double, 9, 1> &state, int armor_id);
    static Eigen::Matrix<double, 4, 9> get_jacobian(const Eigen::Matrix<double, 9, 1> &state, int armor_id);

  PredictionResult update_frame(double timestamp_ms,
                               const std::vector<Eigen::Vector4d> &observations, double horizon_ms = 50);
  void predict(double dt);
  void update(const Eigen::Vector4d &z);

  Eigen::Matrix<double, 9, 1> X = Eigen::Matrix<double, 9, 1>::Zero();
  Eigen::Matrix<double, 9, 9> P = (Eigen::Matrix<double, 9, 1>() << 10, 10, 10, 10, 10, 10, 10, 10, .01).finished().asDiagonal();
  Eigen::Matrix<double, 9, 9> Q = (Eigen::Matrix<double, 9, 1>() << .01, .1, .01, .1, .01, .1, .01, .5, .0001).finished().asDiagonal();
  Eigen::Matrix4d R = Eigen::Vector4d(.005, .005, .05, .05).asDiagonal();
  bool is_initialized = false;
private:
  double last_timestamp_ms = -1;
};
