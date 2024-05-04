/**
 * Implementation of KalmanFilter class.
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 */

#include <iostream>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(double time_step, const Eigen::MatrixXd& state_transition,
                           const Eigen::MatrixXd& observation, const Eigen::MatrixXd& process_noise_cov,
                           const Eigen::MatrixXd& measurement_cov, const Eigen::MatrixXd& estimated_cov)
  : state_transition(state_transition)
  , observation(observation)
  , process_noise_cov(process_noise_cov)
  , measurement_cov(measurement_cov)
  , initial_state_covariance(estimated_cov)
  , time_step(time_step)
  , initialized(false)
  , identity(state_transition.rows(), state_transition.rows())
  , predicted_state(state_transition.rows())
  , estimated_state(state_transition.rows())
{
  identity.setIdentity();
}

KalmanFilter::KalmanFilter()
{
}

void KalmanFilter::init(double initial_time, const Eigen::VectorXd& initial_state)
{
  predicted_state = initial_state;
  estimated_cov = initial_state_covariance;
  this->initial_time = initial_time;
  current_time = initial_time;
  initialized = true;
}

void KalmanFilter::init()
{
  predicted_state.setZero();
  estimated_cov = initial_state_covariance;
  initial_time = 0;
  current_time = initial_time;
  initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd& process_noise)
{
  if (!initialized)
    throw std::runtime_error("Filter is not initialized!");

  estimated_state = state_transition * predicted_state;
  estimated_cov = state_transition * estimated_cov * state_transition.transpose() + process_noise_cov;  
  kalman_gain = estimated_cov * observation.transpose() *
                (observation * estimated_cov * observation.transpose() + measurement_cov).inverse();
  estimated_state += kalman_gain * (process_noise - observation * estimated_state);
  estimated_cov = (identity - kalman_gain * observation) * estimated_cov;
  predicted_state = estimated_state;
  current_time += time_step;
}

void KalmanFilter::update(const Eigen::VectorXd& process_noise, double time_step,
                          const Eigen::MatrixXd state_transition)
{
  this->state_transition = state_transition;
  this->time_step = time_step;
  update(process_noise);
}