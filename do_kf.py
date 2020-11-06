"""Performs the Kalman filtering"""
import json

from pykalman import KalmanFilter
import numpy as np

import util

POSITION_SENSOR_READINGS = "position_sensor_readings.txt"
VELOCITY_SENSOR_READINGS = "velocity_sensor_readings.txt"
KALMAN_OUTPUT_FILENAME = "kalman_output.txt"
KALMAN_SETTINGS = "settings.json"

def form_transition_matrix(dt):
  """Forms the transition matrix given change in time"""
  tm = np.asarray([
    [1, 0, dt,   0],
    [0, 1,  0,  dt],
    [0, 0,  1,   0],
    [0, 0,  0,   1]
  ])
  return tm

## Read in all of our information.
sensor_position = util.read_data(POSITION_SENSOR_READINGS)
sensor_velocity = util.read_data(VELOCITY_SENSOR_READINGS)
with open(KALMAN_SETTINGS, 'r') as f:
  kalman_settings = json.load(f)

## Specify all of the Kalman Filter internals.
transition_matrix = form_transition_matrix(0.1)
initial_transition_covariance = np.eye(4) * 0.1 

# Since we're using a positional/velocity system/model, we can measure our state variables directly.
# Thus our observation matrix doesn't have to transform data at all.
observation_matrix = np.asarray([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
])

sensor_pos_sigma = kalman_settings["sensors"]["position_sensor"]["sigma"]
sensor_vel_sigma = kalman_settings["sensors"]["velocity_sensor"]["sigma"]

# Dividing by 1000 here to convert to seconds from milliseconds
observation_covariance = np.asarray([
  [sensor_pos_sigma,                0,                0,                0],
  [               0, sensor_pos_sigma,                0,                0],
  [               0,                0, sensor_vel_sigma,                0],
  [               0,                0,                0, sensor_vel_sigma]
])

initial_state = np.asarray([
  sensor_position[0, 1], 
  sensor_position[0, 2], 
  0, 
  0
])
initial_state_covariance = np.asarray([
  [sensor_pos_sigma,                0,                0,                0],
  [               0, sensor_pos_sigma,                0,                0], 
  [               0,                0, sensor_vel_sigma,                0],
  [               0,                0,                0, sensor_vel_sigma]
])

## Initialize the Kalman Filter.
print(f"Transition matrix dimensions: {transition_matrix.shape}")
print(f"Observation matrix dimensions: {observation_matrix.shape}")
print(f"Initial transition covariance dimensions: {initial_transition_covariance.shape}")
print(f"Observation covariance dimensions: {observation_covariance.shape}")
print(f"Initial state dimensions: {initial_state.shape}")
print(f"Initial state covariance dimensions: {initial_state_covariance.shape}")
kf = KalmanFilter(
  transition_matrices=transition_matrix,
  observation_matrices=observation_matrix,
  transition_covariance=initial_transition_covariance,
  observation_covariance=observation_covariance,
  transition_offsets=None,
  observation_offsets=None,
  initial_state_mean=initial_state,
  initial_state_covariance=initial_state_covariance
)

## Iteratively estimate mean and cov of hidden states.
time_steps = sensor_position.shape[0]
num_state_vars = observation_covariance.shape[0]
state_means = []
state_covs = []

# Do the first time step.
state_means.append(initial_state)
state_covs.append(initial_state_covariance)

# Iterate through the time steps.
for time_step in range(time_steps - 1):
# for time_step in range(100):
  # Get the next time-step's observations.
  observations = np.asarray([
    sensor_position[time_step + 1, 1],
    sensor_position[time_step + 1, 2],
    sensor_velocity[time_step + 1, 1],
    sensor_velocity[time_step + 1, 2]
  ])

  dt = sensor_position[time_step + 1, 0] - sensor_position[time_step, 0]

  new_state_mean, new_state_cov = kf.filter_update(
    filtered_state_mean=state_means[time_step],
    filtered_state_covariance=state_covs[time_step],
    observation=observations,
    transition_matrix=form_transition_matrix(dt)
  )

  state_means.append(new_state_mean)
  state_covs.append(new_state_cov)

# Form the string to write.
# header = "Time, Pos_x_mu, Pos_y_mu, Vel_x_mu, Vel_y_mu, Pos_x_sigma, Pos_y_sigma, Vel_x_sigma, Vel_y_sigma"
# list_to_write = [header]
# # times = sensor_position[:, 0]
# times = sensor_position[:100, 0]
# for time_step in range(times.shape[0]):
#   new_line = [times[time_step], *state_means[time_step], *state_covs[time_step]]
#   new_line = [str(x) for x in new_line]
#   list_to_write.append(', '.join(new_line))

# str_to_write = '\n'.join(list_to_write)

# with open(KALMAN_OUTPUT_FILENAME, 'w') as f:
#   f.write(str_to_write)

np.save("kalman_output_state_means", state_means)
np.save("kalman_output_state_covs", state_covs)