import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

import scipy.ndimage as ndi
from skimage.morphology import watershed, dilation, erosion
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.filters import median, gaussian, sobel
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from joblib import delayed, Parallel
from openslide import OpenSlide
import os
from tqdm import tqdm

def _compute_initial_thresholds(image):
  bin_edges = np.arange(0.0, 256.0, 1.0)
  histogram, bin_edges = np.histogram(image, bins=bin_edges, density=True)
  bin_positions = (bin_edges[:-1] + bin_edges[1:]) / 2
  pwm = np.cumsum(histogram[histogram > 0] * bin_positions[histogram > 0]) / np.cumsum(histogram[histogram > 0])
  pfit = np.polyfit(bin_positions[histogram > 0], pwm, 15)
  polynomial = np.poly1d(pfit)
  candidate_thresholds = [
    root
    for root in polynomial.deriv(3).roots
    if root.imag == 0.0
  ]
  return candidate_thresholds

def _threshold_image(image, thresholds, im_x=None):
  thresholded = np.zeros((len(thresholds), *image.shape), dtype=int)
  median_areas = []
  for idx, threshold in enumerate(thresholds):
    thresholded[idx][image >= threshold] = 1
    thresholded[idx] = label(thresholded[idx])
    regions = regionprops(thresholded[idx], intensity_image=image)
    areas = np.array(list(map(lambda x: x.area, regions)))
    if len(areas) > 0 and areas.max() < image.size / 2:
      median_area = np.median(areas)
      if len(areas) != 1:
        median_areas.append((idx, median_area))
    else:
      median_areas.append((idx, 0))
  idx, median_area = max(median_areas, key=lambda x: x[1])
  return idx, thresholded, regionprops(thresholded[idx], intensity_image=image)

class RegionWrapper(object):
  def __init__(self, region, offset):
    self.xb, self.yb, self.xe, self.ye = offset.bbox
    self.region = region

  @property
  def bbox(self):
    xb, yb, xe, ye = self.region.bbox
    return (xb + self.xb, yb + self.yb, xe + self.xb, ye + self.yb)

  @property
  def area(self):
    return self.region.area

  @property
  def image(self):
    return self.region.image

  @property
  def solidity(self):
    return self.region.solidity

def _area_improvement(image, regions, thresholded, thresholds, level, large=5, small=0.25):
  mean_area = np.array(list(map(lambda x: x.area, regions))).mean()
  is_small = lambda x: x.area < small * mean_area
  is_large = lambda x: x.area > large * mean_area
  
  to_add = []
  final_threshold = np.zeros(image.shape)
  for _, region in enumerate(regions):
    if is_large(region):
      to_refine = [region]
      new_thresh = thresholds[level]
      while not len(to_refine) == 0:
        new_thresh += 1
        new_refine = []
        for _ in range(len(to_refine)):
          parent_region = to_refine.pop()
          xb, yb, xe, ye = parent_region.bbox
          new_thresholded = np.zeros(image[xb:xe, yb:ye].shape)
          new_thresholded[image[xb:xe, yb:ye] > new_thresh] = 1
          new_thresholded = label(new_thresholded)
          regions = regionprops(
            new_thresholded,
            intensity_image=image[xb:xe, yb:ye]
          )
          for region in regions:
            if not is_small(region) and not is_large(region):
              to_add.append(RegionWrapper(region, parent_region))
            elif is_large(region):
              new_refine.append(RegionWrapper(region, parent_region))
        to_refine += new_refine
      for region in to_refine:
        to_add.append(region)
    elif not is_small(region):
      to_add.append(region)
  for region in to_add:
    xb, yb, xe, ye = region.bbox
    final_threshold[xb:xe, yb:ye] += region.image
  final_threshold = label(final_threshold)
  return final_threshold, regionprops(final_threshold, intensity_image=image)

def _watershed(threshold):
  regions = regionprops(threshold)

  complete_watershed = np.zeros(threshold.shape)
  for region in regions:
    xb, yb, xe, ye = region.bbox
    image = region.image
    extent = region.solidity
    characteristic_size = max(int(0.1 * extent * np.sqrt(region.area)), 3)

    distance = ndi.distance_transform_edt(image)
    ddist = distance > extent * distance.max()
    distance[ddist] = distance.max()
    maxima = peak_local_max(
      ddist, indices=False, footprint=np.ones((characteristic_size, characteristic_size)), labels=image)
    markers = label(maxima)
    ws = watershed(-distance, markers, mask=image, watershed_line=True)
    complete_watershed[xb:xe, yb:ye] += ws

  regions = regionprops(label(complete_watershed))
  mean_area = np.array(list(map(lambda x: x.area, regions))).mean()
  is_small = lambda x: x.area < 0.25 * mean_area
  final_threshold = np.zeros(complete_watershed.shape)
  final_threshold_regions = []
  for idx, region in enumerate(regions, 1):
    if not is_small(region):
      xb, yb, xe, ye = region.bbox
      final_threshold_regions.append(region)
      final_threshold[xb:xe, yb:ye] += region.image * idx
  return final_threshold, final_threshold_regions

def MANA(image, large=5, small=0.25, smoothing=2, sep=False):
  """Computes object-regions using the MANA algorithm
    (doi:10.1186/s12938-018-0518-0).
  
  Args:
    image (numpy.array): input image containing cells.

  Returns:
    A segmentation mask delineating all detected cells, and a list of
    region properties for each cell detected.
  """
  image = gaussian(image, smoothing)
  im_x = np.array(image.copy())
  thresholds = _compute_initial_thresholds(image)
  level, thresholded, regions = _threshold_image(image, thresholds, im_x=im_x)
  final_threshold, regions = _area_improvement(image, regions, thresholded, thresholds, level, large=large, small=small)
  watershed_segmentation, watershed_regions = _watershed(final_threshold)
  return watershed_segmentation, watershed_regions

def MANAObjects(image, large=5, small=0.25, smoothing=2, sep=False, size=(64,64)):
  _, reg = MANA(image, large=large, small=small, smoothing=smoothing, sep=sep)
  for region in reg:
    x, y = region.centroid
    xb, yb, xe, ye = (x - size[0] // 2, y - size[0] // 2, x + size[0] // 2, y + size[0] // 2)
    if xb < 0 or yb < 0 or xe >= image.shape[0] or ye >= image.shape[1]:
      continue
    yield (x, y), image[int(xb):int(xe), int(yb):int(ye)]

def _MANAArray_aux(image, x, y, large=5, small=0.25, smoothing=2, sep=False, size=(64,64), verbose=False):
  positions = []
  image = image / (2 ** 16 - 1)
  image = image * (2 ** 8 - 1)
  if image.mean() / 255 > 0.8:
    return positions
  for center, _ in MANAObjects(image, large=large, small=small, smoothing=smoothing, sep=sep, size=size):
    cx, cy = center
    cx += x
    cy += y
    positions.append((cx, cy))
  return positions

def MANAArray(frame, large=5, small=0.25, smoothing=2, sep=False, size=(64,64), verbose=False, n_jobs=8):
  w, h = frame.shape
  steps_x = w // 1000
  steps_x += (steps_x * (2 * size[0])) // 1000
  steps_y = h // 1000
  steps_y += (steps_y * (2 * size[1])) // 1000
  step_iter = (
    (idx, idy)
    for idx in range(steps_x)
    for idy in range(steps_y)
  )
  res = Parallel(n_jobs=n_jobs)(
    delayed(_MANAArray_aux)(
      frame[
        idx * (1000 - 2 * size[0]):idx * (1000 - 2 * size[0]) + 1000,
        idy * (1000 - 2 * size[1]):idy * (1000 - 2 * size[1]) + 1000
      ],
      idx * (1000 - 2 * size[0]), idy * (1000 - 2 * size[1]),
      large=large, small=small, smoothing=smoothing, sep=sep, size=size, verbose=verbose
    )
    for idx, idy in tqdm(step_iter, total=steps_x*steps_y)
  )
  return (
    (elem, (elem[0] - size[0] // 2, elem[1] - size[1] // 2))
    for array in res
    for elem in array
  )
