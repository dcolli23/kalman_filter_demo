"""Generates noisy sensor data given raw mouse input"""
import json
import numpy as np
import random

INPUT_FILENAME = "time_coord_output.txt"
KALMAN_SETTINGS = "settings.json"
VELOCITY_OUTPUT_FILENAME = "velocity_sensor_readings.txt"
POSITION_OUTPUT_FILENAME = "position_sensor_readings.txt"

# Read in the raw mouse recording.
with open(INPUT_FILENAME, 'r') as f:
  inputs = f.read().split('\n')

inputs = [[float(j) for j in i.replace('(', '').replace(')','').split(', ')] for i in inputs]


# Read in the sensor settings.
with open(KALMAN_SETTINGS, 'r') as f:
  kalman_settings = json.load(f)

pos_mu = kalman_settings["sensors"]["position_sensor"]["mu"]
pos_sigma = kalman_settings["sensors"]["position_sensor"]["sigma"]
vel_mu = kalman_settings["sensors"]["velocity_sensor"]["mu"]
vel_sigma = kalman_settings["sensors"]["velocity_sensor"]["sigma"]

# Generate the position readings.
position_sensor_readings = []
for time, pos_x, pos_y in inputs:
  # Generate random positions based on position sensor noise.
  noisy_x = random.gauss(mu=pos_mu, sigma=pos_sigma) + pos_x
  noisy_y = random.gauss(mu=pos_mu, sigma=pos_sigma) + pos_y

  position_sensor_readings.append((time, noisy_x, noisy_y))

with open(POSITION_OUTPUT_FILENAME, 'w') as f:
  f.write('\n'.join([str(x) for x in position_sensor_readings]))

# Generate the velocity readings.
# NOTE: I'm just going to use a centered finite difference here. It doesn't have to be too accurate.
# NOTE: We're also making an assumption here that our time steps are similar enough such that we can
#       use a finite difference formula which requires that the steps actually be the same.
velocity_sensor_readings = [(inputs[0][0], 0, 0)] # start with zero so data is same length.
for time_point in range(1, len(inputs[1:-1]) + 1):
  readings_t_plus_1 = inputs[time_point + 1]
  readings_t = inputs[time_point]
  readings_t_minus_1 = inputs[time_point - 1]
  
  t_range = readings_t_plus_1[0] - readings_t_minus_1[0]
  x_range = readings_t_plus_1[1] - readings_t_minus_1[1]
  y_range = readings_t_plus_1[2] - readings_t_minus_1[2]

  vel_x_truth = x_range / t_range
  vel_y_truth = y_range / t_range

  vel_x_noisy = vel_x_truth + random.gauss(mu=vel_mu, sigma=vel_sigma)
  vel_y_noisy = vel_y_truth + random.gauss(mu=vel_mu, sigma=vel_sigma)

  velocity_sensor_readings.append((readings_t[0], vel_x_noisy, vel_y_noisy))

# And we put one last reading in so that our data is the same shape.
velocity_sensor_readings.append((inputs[-1][0], 0, 0))

# Now we write out velocity output.
with open(VELOCITY_OUTPUT_FILENAME, 'w') as f:
  f.write('\n'.join([str(x) for x in velocity_sensor_readings]))