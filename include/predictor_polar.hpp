#pragma once
#include <Eigen/Dense>

using PolarState = Eigen::Matrix<double, 9, 1>;
using PolarCovariance = Eigen::Matrix<double, 9, 9>;
using PolarObservation = Eigen::Vector4d;
using PolarJacobian = Eigen::Matrix<double, 4, 9>;
using PolarGain = Eigen::Matrix<double, 9, 4>;
constexpr double polar_pi = 3.14159265358979323846;

// State only; all operations are ordinary functions below.
struct PolarEKF
{
  PolarState X = PolarState::Zero();
  PolarCovariance F = PolarCovariance::Identity();
  PolarCovariance P = (PolarState() << 10, 10, 10, 10, 10, 10, 10, 10, .01).finished().asDiagonal();
  PolarCovariance Q = (PolarState() << .01, .1, .01, .1, .01, .1, .01, .5, .0001).finished().asDiagonal();
  Eigen::Matrix4d R = PolarObservation(.005, .005, .05, .05).asDiagonal();
  bool is_initialized = false;
};

double polar_wrap_to_pi(double a);
PolarObservation polar_angular_residual(const PolarObservation &a, const PolarObservation &b);
int polar_round_even(double x);
void polar_predict(PolarEKF &filter, double dt);
PolarObservation polar_h(const PolarState &s, int plate_idx);
PolarJacobian polar_get_jacobian(const PolarState &s, int id);
void polar_update(PolarEKF &filter, const PolarObservation &z);
