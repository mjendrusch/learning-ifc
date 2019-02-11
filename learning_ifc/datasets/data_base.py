from enum import Enum
from random import Random

class DataMode(Enum):
  """Dataset split type, indicating training, validation or testing."""
  TRAIN = 0
  VALID = 1
  TEST = 2

def random_split(data, seed=None, split=None):
  """Performs a random split on a given dataset.

  Args:
    data (array-like): data to perform a random split on.
    seed (int): seed to perform random splits with.
    split (dict): dictionary specifying percentages per part
      to be included into the split, as well as the part names.

  Returns:
    Dictionary containing the random split by part.
  """
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
    offset = int(data.size(0) * split[piece])
    stop = start + offset
    result[piece] = data[start:stop]
    start += offset
  return result
