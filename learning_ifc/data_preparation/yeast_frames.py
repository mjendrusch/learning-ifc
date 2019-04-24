"""Contains helpers for IFC frame processing."""

import av
import numpy as np
from scipy.ndimage.filters import convolve
from matplotlib import pyplot as plt

def is_yeast_frame(frame):
  """Checks, whether an image contains at least
  one yeast cell.

  Args:
    frame (numpy.array): frame to check for yeast in.
  """
  frame = frame.astype(np.float)[20:]
  frame = frame - frame.min()
  frame = frame / frame.max()

  E = convolve(frame, np.ones((21, 21)) / (21 ** 2), mode='mirror')
  var = convolve((frame - E) ** 2, np.ones((21, 21)) / (21 ** 2), mode='mirror')
  maxvar = var.max()
  minvar = var.mean()
  return maxvar > 2 * minvar

def crop_yeast(frame):
  """Extracts a list of 128x128 crops containing all
  yeast cells in a frame.

  Args:
    frame (numpy.array): frame to extract yeast from.
  """
  frame = frame.astype(np.float)[:, 20:]
  frame = frame - frame.min()
  frame = frame / frame.max()

  E = convolve(frame, np.ones((21, 21)) / (21 ** 2), mode='mirror')
  var = convolve((frame - E) ** 2, np.ones((21, 21)) / (21 ** 2), mode='mirror')
  amx = np.argmax(var)
  cx, cy = amx // frame.shape[0], amx % frame.shape[1]
  start = max(0, cy + 20 - 64)
  stop = min(start + 128, frame.shape[1]) 
  crop = frame[:, start:stop]
  return crop
