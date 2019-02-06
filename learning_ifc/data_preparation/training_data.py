import sys
from itertools import islice

from skimage.external.tifffile import TiffWriter
from pims import ND2_Reader

from learning_ifc.segmentation import MANAArray

path = sys.argv[1]

with ND2_Reader(path) as frames:
  frames.iter_axes = 'z'
  frames.bundle_axes = 'xy'
  fr = frames[21] # maximally in-focus frame

  yeasts = [
    (slice(int(x), int(x) + 128), slice(int(y), int(y) + 128))
    for _, (x, y) in islice(MANAArray(fr, size=(128, 128)), 1000)
  ]

  for level, frame in enumerate(frames):
    for cell, (xs, ys) in enumerate(yeasts):
      with TiffWriter(path + f".cell_{cell}.tif", append=True) as tiff:
        yeast = frame[xs, ys]
        tiff.save(yeast)
