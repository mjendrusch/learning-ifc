from enum import Enum
from random import Random

class DataMode(Enum):
  TRAIN = 0
  VALID = 0
  TEST = 0

def random_split(data, seed=None, split=None):
  if split is None:
    split = {
      DataMode.TRAIN: 0.7,
      DataMode.VALID: 0.2,
      DataMode.TEST: 0.1
    }
  random = Random()
  random.seed(seed)
  random.shuffle(data)
  result = {}
  start = 0
  for piece in split:
    offset = int(len(data) * split)
    stop = start + int(len(data) * split)
    result[piece] = data[start:stop]
    start += offset
  return result
