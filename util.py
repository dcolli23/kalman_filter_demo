import numpy as np

def read_data(file_name):
  """Reads in data in txt file

  NOTE: I know this would be better to do with np.load/save but I want the option to expand to C++
  eventually so I'm keeping in simple txt files.
  """
  with open(file_name, 'r') as f:
    reading = f.read().split('\n')
  reading = np.asarray([
    [float(j) for j in i.replace('(','').replace(')','').split(', ')] 
    for i in reading
  ])
  return reading