import os
import numpy as np
import pims
from PIL import Image, ImageSequence
import torch
from torch.utils.data import Dataset
from learning_ifc.datasets.data_base import DataMode, random_split

class BrightfieldStacks(Dataset):
  """Dataset of brightfield yeast stacks for multiple strains at 41 consecutive z-positions
    separated by 0.25 µm each, centered around the yeast focal plane.
  """

  def __init__(self, transform=lambda x: x, data_mode=DataMode.VALID, split_seed=123456):
    """Dataset of brightfield yeast stacks for multiple strains at 41 consecutive z-positions
    separated by 0.25 µm each, centered around the yeast focal plane.

    Args:
      transform (callable): transforms to apply to datapoints.
      data_mode (DataMode): part of split to use (one of TRAIN, VALID, TEST).
      split_seed (int): seed to use for random split generation per class.
    """
    super(BrightfieldStacks, self).__init__()
    self.split_seed = split_seed
    self.data_mode = data_mode
    self.transform = transform
    self.classes = np.array(("cerevisiae", "ludwigii", "pombe"))
    self.class_count = 1000
    base_data = self._load_data()
    split = {}
    for idx, data_source in enumerate(base_data):
      data_split = random_split(data_source, idx * split_seed)
      for name in data_split:
        split[name] = (
          data_split[name]
          if name not in split
          else torch.cat((split[name], data_split[name]), dim=0)
        )
    self.train = split[DataMode.TRAIN]
    self.test = split[DataMode.TEST]
    self.valid = split[DataMode.VALID]

  def _load_data(self):
    result = []
    path = os.path.dirname(os.path.realpath(__file__))
    path = "/".join(path.split("/")[:-1]) + "/stacks/brightfield/"
    for name in self.classes:
      strain_result = []
      stack_path = f"{path}/{name}/"
      for root, _, files in os.walk(stack_path):
        for filename in files:
          frames = []
          with Image.open(f"{root}/{filename}") as stack:
            for frame in ImageSequence.Iterator(stack):
              frames.append(torch.Tensor(np.array(frame).astype(float)).unsqueeze(0))
          tensor = torch.cat(frames, dim=0).unsqueeze(0)
          strain_result.append(tensor)
      strain_result = torch.cat(strain_result, dim=0)
      result.append(strain_result)
    return result

  def __getitem__(self, idx):
    if self.data_mode == DataMode.TRAIN:
      result = self.train
    elif self.data_mode == DataMode.TEST:
      result = self.test
    elif self.data_mode == DataMode.VALID:
      result = self.valid
    return (
      self.transform(result[idx]),
      idx // (len(result) // len(self.classes))
    )

  def __len__(self):
    if self.data_mode == DataMode.TRAIN:
      result = self.train
    elif self.data_mode == DataMode.TEST:
      result = self.test
    elif self.data_mode == DataMode.VALID:
      result = self.valid
    return result.size(0)

class Brightfield(BrightfieldStacks):
  """Dataset of brightfield yeast images for multiple strains at 41 consecutive z-positions
  separated by 0.25 µm each, centered around the yeast focal plane. Stacks are split up into
  separate images annotated with strain and distance from focal plane.
  """
  def __init__(self, transform=lambda x: x,
               focus_transform=lambda x: x,
               data_mode=DataMode.VALID,
               split_seed=123456):
    """Dataset of brightfield yeast images for multiple strains at 41 consecutive z-positions
    separated by 0.25 µm each, centered around the yeast focal plane. Stacks are split up into
    separate images annotated with strain and distance from focal plane.

    Args:
      transform (callable): transforms to apply to datapoints.
      focus_transform (callable): transforms to apply to distance to focal plane.
      data_mode (DataMode): part of split to use (one of TRAIN, VALID, TEST).
      split_seed (int): seed to use for random split generation per class.
    """
    super(Brightfield, self).__init__(
      transform=transform,
      data_mode=data_mode,
      split_seed=split_seed
    )
    self.focus_transform = focus_transform
    self._unstack_data()

  def _unstack_data(self):
    self.train = self.train.reshape(
      self.train.size(0) * self.train.size(1),
      1,
      self.train.size(2),
      self.train.size(3)
    )
    self.test = self.test.reshape(
      self.test.size(0) * self.test.size(1),
      1,
      self.test.size(2),
      self.test.size(3)
    )
    self.valid = self.valid.reshape(
      self.valid.size(0) * self.valid.size(1),
      1,
      self.valid.size(2),
      self.valid.size(3)
    )

  def __getitem__(self, idx):
    data, strain = super(Brightfield, self).__getitem__(idx)
    focus = torch.tensor(
      [self.focus_transform((idx % 41 - 20) * 0.05)],
      dtype=torch.float32
    )
    return data, strain, focus

class BrightfieldDevice(Dataset):
  def __init__(self, ratio, cells=100, transform=lambda x: x, seed=123456):
    """Dataset simulating a single run of imaging flow cytometry by remixing
    yeast frames from real imaging flow cytometry runs.

    Args:
      ratio (tuple): percentages of each yeast strain in the IFC sequence.
      cells (int): number of frames containing cells in the IFC sequence.
      transform (callable): transforms to apply to datapoints.
      seed (int): seed to use for random IFC run remixing.
    """
    super(BrightfieldDevice, self).__init__()
    self.transform = transform
    self.ratio = ratio
    self.cells = cells
    self.seed = seed
    self.base_data = ...
    self.data = ...
    self.labels = ...
    self._load_data()
    self._sample_run()

  def _load_data(self):
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
        frames = []
        with pims.open(f"{root}/{filename}") as stack:
          for frame in stack:
            if frame.shape == (128, 128):
              frames.append(torch.Tensor(frame).unsqueeze(0))
        tensor = torch.cat(frames, dim=0).unsqueeze(0)
        strain_result[species].append(tensor)
    for name in strain_result:
      strain_result[name] = torch.cat(strain_result[name], dim=0)
    self.base_data = strain_result

  def _sample_run(self):
    result = []
    labels = []
    for idx, name in enumerate(self.base_data):
      split_ratio = self.ratio[idx] * self.cells / len(self.base_data[name])
      take = random_split(
        self.base_data[name],
        seed=idx * self.seed,
        split={
          "accept": split_ratio,
          "reject": 1 - split_ratio
        }
      )["accept"]
      result.append(take)
      labels += torch.Tensor(
        [idx] * take.size(0), dtype="long"
      ).unsqueeze(0).unsqueeze(0)
    result = torch.cat(result, dim=0)
    labels = torch.cat(labels, dim=0)
    self.data = result
    self.labels = labels

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

  def __len__(self):
    return self.cells
