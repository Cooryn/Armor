#pragma once
#include <Eigen/Dense>

constexpr double polar_pi = 3.14159265358979323846;

class PolarEKF
{
public:
  void predict(double dt);
  void update(const Eigen::Vector4d &z);

  Eigen::Matrix<double, 9, 1> X = Eigen::Matrix<double, 9, 1>::Zero();
  Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
  Eigen::Matrix<double, 9, 9> P = (Eigen::Matrix<double, 9, 1>() << 10, 10, 10, 10, 10, 10, 10, 10, .01).finished().asDiagonal();
  Eigen::Matrix<double, 9, 9> Q = (Eigen::Matrix<double, 9, 1>() << .01, .1, .01, .1, .01, .1, .01, .5, .0001).finished().asDiagonal();
  Eigen::Matrix4d R = Eigen::Vector4d(.005, .005, .05, .05).asDiagonal();
  bool is_initialized = false;
};

double polar_wrap_to_pi(double a);
Eigen::Vector4d polar_angular_residual(const Eigen::Vector4d &a, const Eigen::Vector4d &b);
int polar_round_even(double x);
Eigen::Vector4d polar_h(const Eigen::Matrix<double, 9, 1> &s, int plate_idx);
Eigen::Matrix<double, 4, 9> polar_get_jacobian(const Eigen::Matrix<double, 9, 1> &s, int id);
