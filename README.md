# Kalman Filter Demo

## Generate Kalman Filter Data

### Raw Mouse Data

```
python mouse_recorder.py
```

Move the mouse around. X, Y position is recorded every millisecond and written to `time_coord_output.txt`.

### Noisy Sensor Data

```
python generate_sensor_data.py
```

Generates noisy position and velocity sensor data given the raw recorded x,y mouse coordinates in the previous steps.

## Running The Kalman Filter

To run the Kalman filter on the recorded mouse data and the generated noisy data:

```
python do_kf.py
```

## Visualizing the Kalman Filter Prediction

To visualize the Kalman Filter prediction alongside the truth:

```
python visualize_results.py full
```