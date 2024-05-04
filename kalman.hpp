/**
 * Kalman filter implementation using Eigen. Based on the following
 * introductory paper:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 */

#include <Eigen/Dense>

#pragma once

class KalmanFilter
{

public:
  KalmanFilter(
      double time_step,
      const Eigen::MatrixXd &state_transition,
      const Eigen::MatrixXd &observation,
      const Eigen::MatrixXd &process_noise_cov,
      const Eigen::MatrixXd &measurement_cov,
      const Eigen::MatrixXd &estimated_cov);

  /**
   * Create a blank estimator.
   */
  KalmanFilter();

  /**
   * Initialize the filter with initial states as zero.
   */
  void init();

  /**
   * Initialize the filter with a guess for initial states.
   */
  void init(double initial_time, const Eigen::VectorXd &initial_state);

  /**
   * Update the estimated state based on measured values. The
   * time step is assumed to remain constant.
   */
  void update(const Eigen::VectorXd &y);

  /**
   * Update the estimated state based on measured values,
   * using the given time step and dynamics matrix.
   */
  void update(const Eigen::VectorXd &y, double time_step, const Eigen::MatrixXd state_transition);

  /**
   * Return the current state and time.
   */
  Eigen::VectorXd state() { return predicted_state; };
  double time() { return current_time; };


private:
  Eigen::MatrixXd state_transition, observation, process_noise_cov,
      measurement_cov, estimated_cov, kalman_gain, initial_state_covariance;

  double initial_time, current_time;

  double time_step;

  bool initialized;

  Eigen::MatrixXd identity;

  Eigen::VectorXd predicted_state, estimated_state;
};
