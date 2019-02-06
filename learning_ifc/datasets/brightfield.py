import os
import pims
import torch
from torch.utils.data import Dataset
from data_base import DataMode, random_split

class BrightfieldStacks(Dataset):
  def __init__(self, transform=lambda x: x, data_mode=DataMode.VALID, split_seed=123456):
    super(BrightfieldStacks, self).__init__()
    self.base_data = []
    self.split_seed = split_seed
    self.data_mode = data_mode
    self.transform = transform
    self.classes = ("Cerevisiae", "Ludwigii", "Pombe")
    self.class_count = 1000
    self._load_data()
    self.split = {}
    for idx, data_source in enumerate(self.base_data):
      data_split = random_split(data_source, idx * self.split_seed)
      for name in data_split:
        self.split[name] = (
          data_split[name]
          if name not in self.split
          else torch.cat((self.split[name], data_split[name]), dim=0)
        )

  def _load_data(self):
    result = []
    path = os.path.dirname(os.path.realpath(__file__))
    path = path.split("/")[:-1].join("/") + "stacks/brightfield/"
    for name in self.classes:
      strain_result = []
      stack_path = f"{path}/{name}/"
      for root, _, files in os.walk(stack_path):
        for filename in files:
          frames = []
          with pims.open(f"{root}/{filename}") as stack:
            for frame in stack:
              frames.append(torch.Tensor(frame).unsqueeze(0))
          tensor = torch.cat(frames, dim=0).unsqueeze(0)
          strain_result.append(tensor)
      strain_result = torch.cat(strain_result, dim=0)
      result.append(strain_result)
    self.base_data = result

  def __getitem__(self, idx):
    return (
      self.transform(self.split[self.data_mode][idx]),
      idx // (len(self.split[self.data_mode]) // len(self.classes))
    )

  def __len__(self):
    return len(self.split[self.data_mode])

class Brightfield(BrightfieldStacks):
  def __init__(self, transform=lambda x: x,
               focus_transform=lambda x: x,
               data_mode=DataMode.VALID,
               split_seed=123456):
    super(Brightfield, self).__init__(
      transform=transform,
      data_mode=data_mode,
      split_seed=split_seed
    )
    self.focus_transform = focus_transform
    self._unstack_data()

  def _unstack_data(self):
    for name in self.split:
      self.split[name] = self.split[name].reshape(
        self.split[name].size(0) * self.split[name].size(1),
        self.split[name].size(2),
        self.split[name].size(3)
      )

  def __getitem__(self, idx):
    data, strain = super(Brightfield, self).__getitem__(idx)
    focus = self.focus_transform((idx % 41 - 20) * 0.25)
    return data, strain, focus

class BrightfieldDevice(Dataset):
  def __init__(self, ratio, cells=100, transform=lambda x: x, seed=123456):
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
    result = []
    path = os.path.dirname(os.path.realpath(__file__))
    path = path.split("/")[:-1].join("/") + "runs/brightfield/"
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

