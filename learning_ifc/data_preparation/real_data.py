import sys
from itertools import islice

import numpy as np
from skimage.external.tifffile import TiffWriter
import av

from learning_ifc.data_preparation.yeast_frames import is_yeast_frame, crop_yeast

path = sys.argv[1]

container = av.open(path)
stream = container.streams.video[0]

for frame in container.decode(stream):
  frame = np.array(frame.to_image()).transpose(2, 0, 1)[0, :, :]
  if is_yeast_frame(frame):
    crop = crop_yeast(frame)
    crop = crop - crop.min()
    crop = crop / crop.max()
    crop = crop * 255
    with TiffWriter(path + f".yeast_substack.tif", append=True) as tiff:
      tiff.save(crop.astype(np.uint8))
