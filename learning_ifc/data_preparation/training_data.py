"""Preprocesses and segments training slides into 41x128x128 stacks
with 1000 cells per class.
"""

import sys
from itertools import islice

from matplotlib import pyplot as plt

from skimage.external.tifffile import TiffWriter
from pims import ND2_Reader

from learning_ifc.data_preparation.segmentation import MANAArray

path = sys.argv[1]

with ND2_Reader(path) as frames:
  frames.iter_axes = 'z'
  frames.bundle_axes = 'xy'
  fr = frames[0] # maximally out-of-focus frame

  yeasts_raw = [
    (slice(int(x), int(x) + 128), slice(int(y), int(y) + 128))
    for _, (x, y) in MANAArray(fr, size=(128, 128))
  ]

  total = 0
  yeasts = []
  accept = [False]
  def press(event):
    accept[0] = event.key == "enter"
  for cell, (xs, ys) in enumerate(yeasts_raw):
    if total >= 1000:
      break
    fig, ax = plt.subplots()
    ax.imshow(fr[xs, ys])

    fig.canvas.mpl_connect('key_press_event', press)

    plt.show(block=False)
    plt.waitforbuttonpress()
    if accept:
      yeasts.append((xs, ys))
      total += 1

  for level, frame in enumerate(frames):
    for cell, (xs, ys) in enumerate(yeasts):
      with TiffWriter(path + f".cell_{cell}.tif", append=True) as tiff:
        yeast = frame[xs, ys]
        tiff.save(yeast)
