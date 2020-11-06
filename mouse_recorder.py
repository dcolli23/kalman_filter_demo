import pyautogui
import time
print('Press Ctrl-C to quit.')

OUTPUT_FILE = "time_coord_output.txt"

time_list = []
x_coord_list = []
y_coord_list = []

start_time = time.time()
try:
    while True:
        x, y = pyautogui.position()
        time_list.append(time.time() - start_time)
        x_coord_list.append(x)
        y_coord_list.append(y)
        positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
        print(positionStr, end='')
        print('\b' * len(positionStr), end='', flush=True)
        time.sleep(0.001)
except KeyboardInterrupt:
    print('\n')
    with open(OUTPUT_FILE, 'w') as f:
      zipped_list = list(zip(time_list, x_coord_list, y_coord_list))
      str_to_write = '\n'.join([str(x) for x in zipped_list])
      f.write(str_to_write)