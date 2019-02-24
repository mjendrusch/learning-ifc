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
path = "/".join(path.split("/")[:-1]) + "/runs/brightfield/"
strain_result = {
  "cerevisiae": [],
  "ludwigii": [],
  "pombe": []
}

for root, _, files in os.walk(path):
  for filename in files:
    species = filename.split("_")[0]
    train_frames = []
    bad_frames = []
    frames = []
    with Image.open(f"{root}/{filename}") as stack:
      for idx, frame in enumerate(ImageSequence.Iterator(stack)):
        frame = np.array(frame)
        if frame.shape == (150, 128):
          frame = frame[11:139, :]
        frames.append(frame)

    current = 0
    current_frame = frames[current]
    def clicked(event):
      global current
      global current_frame
      if event.button == 1:
        train_frames.append(current)
      if event.button in (2, 3):
        bad_frames.append(current)
      current += 1
      current_frame = frames[current]
      ax.imshow(current_frame); fig.canvas.draw()

    fig = plt.figure(figsize=(1.28, 1.28))
    ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
                      aspect='equal', frameon=False)
    fig.canvas.mpl_connect("button_press_event", clicked)
    ax.imshow(current_frame); plt.show(); fig.canvas.draw()
