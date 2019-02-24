import os
import numpy as np
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
import json

def tile(paths):
  names = [
    [
      paths[idx * 10 + idy]
      for idy in range(10)
    ]
    for idx in range(10)
  ]
  valid_names = []

  def clicked(event):
    x, y = map(lambda x: int(x / 128), (event.xdata, event.ydata))
    if event.button == 1:
      if names[x][y] not in valid_names:
        valid_names.append(names[x][y])

  fig = plt.figure(figsize=(1280, 1280))
  ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
                    aspect='equal', frameon=False)
  for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(plt.NullFormatter())
    axis.set_major_locator(plt.NullLocator())

  col_images = []
  for name_row in names:
    row_images = []
    for path in name_row:
      with Image.open(path) as stack:
        stack.seek(21)
        image = np.array(stack)
        row_images.append(image)
    row = np.concatenate(row_images, axis=0)
    col_images.append(row)
  full = np.concatenate(col_images, axis=1)

  fig.canvas.mpl_connect("button_press_event", clicked)

  ax.imshow(full)
  plt.show()
  return valid_names

path = os.path.dirname(os.path.realpath(__file__))
path = "/".join(path.split("/")[:-1]) + "/stacks/brightfield/"
valid = {}
for name in ("cerevisiae", "ludwigii", "pombe"):
  strain_result = []
  stack_path = f"{path}/{name}/"
  filenames = [
    f"{root}/{filename}"
    for root, _, files in os.walk(stack_path)
    for filename in files
  ]
  filenames.sort()
  valid_names = []
  for batch in range(10):
    subnames = filenames[batch * 100:(batch + 1) * 100]
    valid_names += tile(subnames)
  valid[name] = valid_names

print(json.dumps(valid))
