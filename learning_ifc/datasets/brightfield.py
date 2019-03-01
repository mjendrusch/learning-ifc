import os
import json
import random
import numpy as np
import pims
from PIL import Image, ImageSequence
import torch
from torch.utils.data import Dataset, Subset
from learning_ifc.datasets.data_base import DataMode, random_split

def _focus(data):
  result = []
  for stack in data:
    variances = stack.view(stack.size(0), -1).std(dim=1)
    in_focus = 20#variances.argmin()
    focus = torch.Tensor([
      max(min((idx - float(in_focus)) * 0.05, 1.0), -1.0)
      for idx in range(stack.size(0))
    ])
    result.append(focus.unsqueeze(0))
  return torch.cat(result, dim=0)

class BrightfieldStacks(Dataset):
  """Dataset of brightfield yeast stacks for multiple strains at 41 consecutive z-positions
    separated by 0.25 µm each, centered around the yeast focal plane.
  """

  def __init__(self, transform=lambda x: x, data_mode=DataMode.VALID, split_seed=123456,
               use_hand_validated=False, use_hand_focused=False):
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
    base_data = self._load_data(use_hand_validated, use_hand_focused)
    if use_hand_focused:
      merged_images, merged_foci = base_data
      self.train = merged_images[0]
      self.valid = merged_images[1]
      self.test = merged_images[2]
      self.train_focus = merged_foci[0]
      self.valid_focus = merged_foci[1]
      self.test_focus = merged_foci[2]
    else:
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
      self.train_focus = _focus(self.train)
      self.test_focus = _focus(self.train)
      self.valid_focus = _focus(self.valid)

  def _default_load_data(self, use_hand_validated):
    result = []
    path = os.path.dirname(os.path.realpath(__file__))
    path = "/".join(path.split("/")[:-1]) + "/stacks/brightfield/"
    with open(path + "valid.json") as valid_file:
      valid_dict = json.loads(next(valid_file))
    for name in self.classes:
      strain_result = []
      stack_path = f"{path}/{name}/"
      filenames = [
        f"{root}/{filename}"
        for root, _, files in os.walk(stack_path)
        for filename in files
      ]
      if use_hand_validated:
        filenames = valid_dict[name]
      for filename in filenames:
        frames = []
        with Image.open(filename) as stack:
          for frame in ImageSequence.Iterator(stack):
            frame = np.array(frame)
            frame = frame - frame.min()
            frame = frame / frame.max()
            frame = frame * 255
            frame = frame.astype(np.uint8)

            frame_tensor = torch.Tensor(frame.astype(float)).unsqueeze(0)
            frames.append(frame_tensor)
        tensor = torch.cat(frames, dim=0).unsqueeze(0)
        am = tensor.view(tensor.size(1), -1).std(dim=1).argmin()
        # if 1 <= am < 40:
        strain_result.append(tensor)
      # strain_result = torch.cat(strain_result, dim=0)
      result.append(strain_result)
    minlen = min(map(len, result))
    for idx, elem in enumerate(result):
      if len(elem) > minlen:
        result[idx] = random.sample(elem, k=minlen)
      result[idx] = torch.cat(result[idx], dim=0)
    return result

  def _hand_focused_load_data(self):
    result = []
    path = os.path.dirname(os.path.realpath(__file__))
    path = "/".join(path.split("/")[:-1]) + "/stacks/brightfield/"
    with open(path + "focus.json") as valid_file:
      invalid_dict = json.loads(next(valid_file))
      focus_dict = json.loads(next(valid_file))
    for name in self.classes:
      strain_focus = focus_dict[name]
      strain_result = []
      for image_path in strain_focus:
        focal_planes = strain_focus[image_path]
        if len(focal_planes) == 1:
          focal_plane = focal_planes[0]
          strain_result.append((image_path, focal_plane))
      result.append(strain_result)
    minlen = min(map(len, result))
    merged_images = [None, None, None]
    merged_foci = [None, None, None]
    for idx, elem in enumerate(result):
      if len(elem) > minlen:
        result[idx] = random.sample(elem, k=minlen)
      result[idx] = random_split(result[idx], seed=self.split_seed)
      for idy, split in enumerate(result[idx]):
        split_tensors = []
        split_foci = []
        for image_path, focus in result[idx][split]:
          split_focus = torch.Tensor([
            max(min((idx - float(focus)) * 0.05, 1.0), -1.0)
            for idx in range(41)
          ]).unsqueeze(0)
          frames = []
          with Image.open(image_path) as img:
            for frame in ImageSequence.Iterator(img):
              frame_tensor = torch.Tensor((np.array(frame) // 256).astype(float)).unsqueeze(0)
              frames.append(frame_tensor)
          split_tensor = torch.cat(frames, dim=0).unsqueeze(0)
          split_tensors.append(split_tensor)
          split_foci.append(split_focus)
        split_tensors = torch.cat(split_tensors, dim=0)
        split_foci = torch.cat(split_foci, dim=0)
        merged_images[idy] = (
          split_tensors
          if merged_images[idy] is None
          else torch.cat((merged_images[idy], split_tensors), dim=0)
        )
        merged_foci[idy] = (
          split_foci
          if merged_foci[idy] is None
          else torch.cat((merged_foci[idy], split_foci), dim=0)
        )
    return merged_images, merged_foci

  def _load_data(self, use_hand_validated, use_hand_focused):
    if use_hand_focused:
      result = self._hand_focused_load_data()
    else:
      result = self._default_load_data(use_hand_validated)
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

  def getfocus(self, idx):
    if self.data_mode == DataMode.TRAIN:
      result = self.train_focus
    elif self.data_mode == DataMode.TEST:
      result = self.test_focus
    elif self.data_mode == DataMode.VALID:
      result = self.valid_focus
    return result[idx]

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
    self.focus_only = False
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
    self.train_focus = self.train_focus.reshape(
      self.train_focus.size(0) * self.train_focus.size(1), 1
    )
    self.test_focus = self.test_focus.reshape(
      self.test_focus.size(0) * self.test_focus.size(1), 1
    )
    self.valid_focus = self.valid_focus.reshape(
      self.valid_focus.size(0) * self.valid_focus.size(1), 1
    )

  def __getitem__(self, idx):
    data, strain = super(Brightfield, self).__getitem__(idx)
    focus = self.getfocus(idx)
    result = (data, strain, focus)
    if self.focus_only:
      result = (data, focus)
    return result

class BrightfieldRefocus(BrightfieldStacks):
  def __init__(self, defocus=5, transform=lambda x: x):
    super(BrightfieldRefocus, self).__init__(
      transform=transform,
      data_mode=DataMode.TRAIN
    )
    self.defocus = defocus

  def __getitem__(self, idx):
    starting_plane = random.choice(range(41 - self.defocus))
    transformed = self.transform(self.train[idx])
    return (
      transformed[starting_plane].unsqueeze(0),
      transformed[starting_plane + self.defocus].unsqueeze(0)
    )

class BrightfieldImage(Brightfield):
  def __init__(self, transform=lambda x: x):
    super(BrightfieldImage, self).__init__(
      transform=transform,
      data_mode=DataMode.TRAIN
    )
    self.labels = None

  def __getitem__(self, idx):
    if self.labels is not None:
      label = self.labels[idx]
    else:
      label = 0
    return self.transform(self.train[idx]), label

class BrightfieldClass(Brightfield):
  def __init__(self, transform=lambda x: x,
               focus_transform=lambda x: x,
               data_mode=DataMode.VALID,
               split_seed=123456, bin_width=0.5):
    super(BrightfieldClass, self).__init__(
      transform=transform,
      focus_transform=focus_transform,
      data_mode=data_mode,
      split_seed=split_seed
    )
    self.bin_width = bin_width
    binned = ((self.train_focus + 1) / self.bin_width)
    bins = [0 for _ in range(int(2 / self.bin_width + 1))]
    for focus in self.train_focus:
      ff = int((focus - 1e-6 + 1) / (self.bin_width - 1e-6))
      bins[ff] += 1
    # self.focus_weights = torch.histc(binned, bins=int(2 / self.bin_width + 1), min=0, max=(1 / self.bin_width + 1))
    # self.focus_weights = self.focus_weights.sum(dim=0, keepdim=True) / self.focus_weights
    print(bins)
    self.focus_weights = torch.Tensor([1 / val for val in bins])

  def __getitem__(self, idx):
    *other, focus = super(BrightfieldClass, self).__getitem__(idx)
    focus = int((focus - 1e-6 + 1) / (self.bin_width - 1e-6))
    return (*other, focus)

class BrightfieldFocusDistance(BrightfieldStacks):
  def __init__(self, data_mode=DataMode.VALID,
               split_seed=123456, transform=lambda x: x):
    super(BrightfieldFocusDistance, self).__init__(
      transform=lambda x: x,
      data_mode=data_mode,
      split_seed=split_seed
    )
    self.independent_transform = transform

  def __getitem__(self, raw_idx):
    idx = raw_idx % super(BrightfieldFocusDistance, self).__len__()
    data, label = super(BrightfieldFocusDistance, self).__getitem__(idx)
    first = random.choice(range(41))
    second = random.choice(range(41))
    difference = (second - first) * 0.05
    return torch.cat((
      self.independent_transform(data[first].unsqueeze(0)),
      self.independent_transform(data[second].unsqueeze(0))
    ), dim=0), torch.tensor([difference], dtype=torch.float)
  
  def __len__(self):
    return 41 ** 2 * super(BrightfieldFocusDistance, self).__len__()

class BrightfieldShot(Brightfield):
  def __init__(self, data_mode=DataMode.VALID, n_shot=1,
               split_seed=123456, transform=lambda x: x):
    super(BrightfieldShot, self).__init__(
      transform=lambda x: x,
      data_mode=data_mode,
      split_seed=split_seed
    )
    self.n_shot = n_shot
    self.independent_transform = transform

  def __getitem__(self, raw_idx):
    idx = raw_idx % super(BrightfieldShot, self).__len__()
    data, label, *_ = super(BrightfieldShot, self).__getitem__(idx)
    data_p, label_p, *_ = super(BrightfieldShot, self).__getitem__(
      random.choice(range(super(BrightfieldShot, self).__len__()))
    )
    return torch.cat((
      self.independent_transform(data),
      self.independent_transform(data_p)
    ), dim=0), int(label == label_p)

class BrightfieldShotFocus(BrightfieldStacks):
  def __init__(self, data_mode=DataMode.VALID, n_shot=1,
               split_seed=123456, transform=lambda x: x):
    super(BrightfieldShotFocus, self).__init__(
      transform=lambda x: x,
      data_mode=data_mode,
      split_seed=split_seed
    )
    self.n_shot = n_shot
    self.independent_transform = transform

  def __getitem__(self, raw_idx):
    idx = raw_idx % super(BrightfieldShotFocus, self).__len__()
    data, label, *_ = super(BrightfieldShotFocus, self).__getitem__(idx)
    data_p, label_p, *_ = super(BrightfieldShotFocus, self).__getitem__(
      random.choice(range(super(BrightfieldShotFocus, self).__len__()))
    )
    return torch.cat((
      self.independent_transform(data[21].unsqueeze(0)),
      self.independent_transform(data_p[21].unsqueeze(0))
    ), dim=0), int(label == label_p)

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
        with Image.open(f"{root}/{filename}") as stack:
          for frame in ImageSequence.Iterator(stack):
            frame = np.array(frame)
            if frame.shape == (150, 128):
              frame = frame[11:139, :]
            if frame.shape == (128, 128):
              frames.append(torch.Tensor(frame).unsqueeze(0).unsqueeze(0))
        tensor = torch.cat(frames, dim=0)
        strain_result[species].append(tensor)
    print(strain_result)
    for name in strain_result:
      print(name)
      print(*map(lambda x: x.size(), strain_result[name]))
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
      labels += [idx] * take.size(0)
    result = torch.cat(result, dim=0)
    labels = torch.tensor(labels, dtype=torch.long).unsqueeze(1).unsqueeze(2)
    self.data = result
    self.labels = labels

  def __getitem__(self, idx):
    return self.transform(self.data[idx]), self.labels[idx]

  def __len__(self):
    return self.cells

class BrightfieldDeviceValid(Dataset):
  def __init__(self, ratio, cells=100, transform=lambda x: x, seed=123456):
    """Dataset simulating a single run of imaging flow cytometry by remixing
    yeast frames from real imaging flow cytometry runs.

    Args:
      ratio (tuple): percentages of each yeast strain in the IFC sequence.
      cells (int): number of frames containing cells in the IFC sequence.
      transform (callable): transforms to apply to datapoints.
      seed (int): seed to use for random IFC run remixing.
    """
    super(BrightfieldDeviceValid, self).__init__()
    self.transform = transform
    self.ratio = ratio
    self.cells = cells
    self.seed = seed
    self.class_names = ("cerevisiae", "ludwigii", "pombe")
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
        with Image.open(f"{root}/{filename}") as stack:
          for frame in ImageSequence.Iterator(stack):
            frame = np.array(frame)
            if frame.shape == (150, 128):
              frame = frame[11:139, :]
            if frame.shape == (128, 128):
              frames.append(torch.Tensor(frame.astype(float)).unsqueeze(0).unsqueeze(0))
        tensor = torch.cat(frames, dim=0)
        strain_result[species].append(tensor)
    print(strain_result)
    for name in strain_result:
      print(name)
      print(*map(lambda x: x.size(), strain_result[name]))
      strain_result[name] = torch.cat(strain_result[name], dim=0)
    self.base_data = strain_result

  def _sample_run(self):
    result = []
    labels = []
    for idx, name in enumerate(self.class_names):
      indices = random.choices(range(len(self.base_data[name])), k=1000)
      result.append(self.base_data[name][indices])
      labels += [idx] * 1000
    result = torch.cat(result, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    focus = torch.zeros(result.size(0), dtype=torch.long)
    self.data = result
    self.labels = labels
    self.focus = focus

  def __getitem__(self, idx):
    return (self.transform(self.data[idx]), self.labels[idx], self.focus[idx])

  def __len__(self):
    return 3000#len(self.data)

class BrightfieldDeviceTrain(Dataset):
  def __init__(self, ratio, cells=100, transform=lambda x: x, seed=123456):
    """Dataset simulating a single run of imaging flow cytometry by remixing
    yeast frames from real imaging flow cytometry runs.

    Args:
      ratio (tuple): percentages of each yeast strain in the IFC sequence.
      cells (int): number of frames containing cells in the IFC sequence.
      transform (callable): transforms to apply to datapoints.
      seed (int): seed to use for random IFC run remixing.
    """
    super(BrightfieldDeviceTrain, self).__init__()
    self.valid = BrightfieldDeviceValid(ratio, cells=cells, transform=transform, seed=seed)
    self.subset = Subset(self.valid, list(range(100)) + list(range(800, 900)) + list(range(1200, 1300)))

  def __getitem__(self, idx_raw):
    idx = idx_raw % len(self.subset)
    return self.subset[idx]

  def __len__(self):
    return 20000

class BrightfieldDeviceImage(Dataset):
  def __init__(self, transform=lambda x: x, seed=123456):
    """Dataset simulating a single run of imaging flow cytometry by remixing
    yeast frames from real imaging flow cytometry runs.

    Args:
      ratio (tuple): percentages of each yeast strain in the IFC sequence.
      cells (int): number of frames containing cells in the IFC sequence.
      transform (callable): transforms to apply to datapoints.
      seed (int): seed to use for random IFC run remixing.
    """
    super(BrightfieldDeviceImage, self).__init__()
    self.labels = None
    self.valid = BrightfieldDeviceValid((0.2, 0.2, 0.6), cells=100, transform=transform, seed=seed)

  def __getitem__(self, idx_raw):
    idx = idx_raw % len(self.valid)
    if self.labels is not None:
      label = self.labels[idx]
    else:
      label = 0
    return self.valid[idx][0], label

  def __len__(self):
    return len(self.valid)
