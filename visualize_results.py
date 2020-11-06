import sys

import imageio
import numpy as np

import util

POSITION_READINGS = "position_sensor_readings.txt"
KALMAN_OUTPUT_MEANS = "kalman_output_state_means.npy"
KALMAN_OUTPUT_COVS = "kalman_output_state_covs.npy"

BOUND_X_MIN = 0
BOUND_X_MAX = 1919
BOUND_Y_MIN = 0
BOUND_Y_MAX = 1079

def get_bounded_min(array, cutoff):
  """Bounds min(`array`) to `cutoff` if min(`array`) < `cutoff`"""
  min_num = np.min(array)
  min_num = cutoff if min_num < cutoff else min_num
  return min_num

def get_bounded_max(array, cutoff):
  """Bounds max(`array`) to `cutoff` if max(`array`) > `cutoff`"""
  max_num = np.min(array)
  max_num = cutoff if max_num > cutoff else max_num
  return max_num

def mark_blob(img, pt_x, pt_y, blob_width, channel):
  """Marks a blob on the image at the indicated point with given width"""
  x_min = pt_x - blob_width // 2
  x_min = 0 if x_min < 0 else x_min
  x_min = img.shape[1] - 1 if x_min >= img.shape[1] else x_min
  x_max = pt_x + blob_width // 2
  x_max = 0 if x_max < 0 else x_max
  x_max = img.shape[1] - 1 if x_max >= img.shape[1] else x_max

  y_min = pt_y - blob_width // 2
  y_min = 0 if y_min < 0 else y_min
  y_min = img.shape[0] - 1 if y_min >= img.shape[0] else y_min
  y_max = pt_y + blob_width // 2
  y_max = 0 if y_max < 0 else y_max
  y_max = img.shape[0] - 1 if y_max >= img.shape[0] else y_max

  img[y_min:y_max, x_min:x_max, channel] = 255
  for i in [0, 1, 2]:
    if i != channel:
      img[y_min:y_max, x_min:x_max, i] = 0

def write_just_position_sensor_readings():
  """Plots and writes just the position sensor reading"""
  positions = util.read_data(POSITION_READINGS)

  times = positions[:, 0]
  pos_x = positions[:, 1]
  pos_y = positions[:, 2]

  min_x = get_bounded_min(pos_x, BOUND_X_MIN)
  min_y = get_bounded_min(pos_y, BOUND_Y_MIN)

  print(f"(X min, Y min) = ({min_x}, {min_y})")
  print(f"(X max, Y max) = ({np.max(pos_x)}, {np.max(pos_y)})")

  # Figure out what our sampling frequency should be.
  pos_sensor_sampling_frequency = 1000 # Hz
  gif_sampling_frequency = 24 # Hz I think?
  pos_reading_sampling_frequency = int(pos_sensor_sampling_frequency / gif_sampling_frequency)

  image = np.ones((BOUND_Y_MAX, BOUND_X_MAX, 3), dtype=np.uint8) * 255

  with imageio.get_writer('visualized_position_reading.gif', mode='I') as writer:
    for time_point in range(pos_x.shape[0] // pos_reading_sampling_frequency):
      time_point = time_point * pos_reading_sampling_frequency
      print(f"Writing time point #{time_point}")
      point_x = pos_x[time_point]
      point_y = pos_y[time_point]    
      
      # Convert to integers.
      point_x = int(point_x)
      point_y = int(point_y)

      # TODO: optimize this by just doing the bounding once and using numpys slicing syntax to mark 
      #       the image.
      # for i in range(-2, 3):
      #   for j in range(-2, 3):
      #     # Bound the sensor readings to our image bounds.
      #     mark_x = point_x + i
      #     mark_y = point_y + j
      #     mark_x = BOUND_X_MIN if mark_x < 0 else mark_x
      #     mark_y = BOUND_Y_MIN if mark_y < 0 else mark_y

      #     mark_x = BOUND_X_MAX if mark_x > BOUND_X_MAX else mark_x
      #     mark_y = BOUND_Y_MAX if mark_y > BOUND_Y_MAX else mark_y

      #     image[mark_y, mark_x, 0] = 255
      #     image[mark_y, mark_x, 1] = 0
      #     image[mark_y, mark_x, 2] = 0
      mark_blob(image, point_x, point_y, 4, 0)

      writer.append_data(image)

def write_full_kalman_prediction_with_truth():
  """Writes the full kalman prediction with the truth values"""
  state_means = np.load(KALMAN_OUTPUT_MEANS)
  state_covs = np.load(KALMAN_OUTPUT_COVS)
  print(state_covs.shape)

  pos_x_predicted_arr = state_means[:, 0]
  pos_y_predicted_arr = state_means[:, 1]
  pos_x_predicted_variance_arr = state_covs[:, 0, 0]
  pos_y_predicted_variance_arr = state_covs[:, 1, 1]

  pos_truth = util.read_data("time_coord_output.txt").astype(np.int)

  # Figure out what our sampling frequency should be.
  pos_sensor_sampling_frequency = 1000 # Hz
  # pos_sensor_sampling_frequency = 24
  gif_sampling_frequency = 24 # Hz I think?
  result_sampling_frequency = int(pos_sensor_sampling_frequency / gif_sampling_frequency)

  image = np.ones((BOUND_Y_MAX, BOUND_X_MAX, 3), dtype=np.uint8) * 255

  with imageio.get_writer('visualized_kalman_result.gif', mode='I') as writer:
    for time_point in range(pos_x_predicted_arr.shape[0] // result_sampling_frequency):
      time_point = time_point * result_sampling_frequency
      print(f"Writing time point #{time_point}")

      pos_x_predicted = int(pos_x_predicted_arr[time_point])
      pos_y_predicted = int(pos_y_predicted_arr[time_point])

      print(f"\tPredicted point (x,y) = ({pos_x_predicted}, {pos_y_predicted})")

      # Mark the predicted value.
      mark_blob(image, pos_x_predicted, pos_y_predicted, 6, 0)
      
      # Mark the actual value.
      mark_blob(image, pos_truth[time_point, 1], pos_truth[time_point, 2], 6, 2)
      
      writer.append_data(image)



for i, arg in enumerate(sys.argv):
  if (arg=="full"):
    write_full_kalman_prediction_with_truth()
    quit()
  elif (arg=="reading"):
    write_just_position_sensor_readings()
    quit()