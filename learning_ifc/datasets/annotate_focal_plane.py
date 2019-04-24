"""Script to annotate cell focal plane."""

import os
import numpy as np
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
import json

def tile(paths):
  
  invalid = []
  focal_planes = {}

  def clicked(event):
    x, y = map(lambda x: int(x / 128), (event.xdata, event.ydata))
    if event.button == 1:
      if paths[y] not in focal_planes:
        focal_planes[paths[y]] = []  
      focal_planes[paths[y]].append(x)
    if event.button in (2, 3):
      if paths[y] not in invalid:
        invalid.append(paths[y])

  fig = plt.figure(figsize=(128 * 20, 128 * 41))
  ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
                    aspect='equal', frameon=False)
  for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(plt.NullFormatter())
    axis.set_major_locator(plt.NullLocator())

  col_images = []
  for image_path in paths:
    row_images = []
    with Image.open(image_path) as stack:
      for image in ImageSequence.Iterator(stack):
        image = np.array(stack)
        row_images.append(image)
    row = np.concatenate(row_images, axis=1)
    col_images.append(row)
  full = np.concatenate(col_images, axis=0)

  fig.canvas.mpl_connect("button_press_event", clicked)

  ax.imshow(full)
  plt.show()
  return invalid, focal_planes

path = os.path.dirname(os.path.realpath(__file__))
path = "/".join(path.split("/")[:-1]) + "/stacks/brightfield/"
invalid = {}
focal = {}
for name in ("cerevisiae", "ludwigii", "pombe"):
  strain_result = []
  stack_path = f"{path}/{name}/"
  filenames = [
    f"{root}/{filename}"
    for root, _, files in os.walk(stack_path)
    for filename in files
  ]
  filenames.sort()
  invalid_names = []
  fps = {}
  for batch in range(50):
    subnames = filenames[batch * 20:(batch + 1) * 20]
    ii, fp = tile(subnames)
    invalid_names += ii
    fps.update(fp)
  invalid[name] = invalid_names
  focal[name] = fps

print(json.dumps(invalid))
print(json.dumps(focal))
