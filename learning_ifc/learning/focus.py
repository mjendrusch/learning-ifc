"""Siamese neural network training for yeast z-distance regression."""
from argparse import ArgumentParser
from copy import copy

import torch.nn as nn

from torchsupport.data.io import netwrite
from torchsupport.data.transforms import \
  Rotation4, Compose, Shift, Zoom, Perturb, Normalize, Translation
from torchsupport.training.training import SupervisedTraining

from learning_ifc.learning.models.compact import Compact, Perceptron, MLP, Siamese
from learning_ifc.datasets.brightfield import BrightfieldFocusDistance, DataMode

def parse_args():
  """Parses neural network training arguments."""
  parser = ArgumentParser(description="Supervisied classification training.")
  parser.add_argument("--test", dest="test", action="store_true")
  parser.set_defaults(test=False)
  parser.add_argument("--testdevice", type=str, default=None)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--task", type=int, default=2)
  parser.add_argument("--net", type=str, default="compact:6:1:256:2")
  parser.add_argument("--classifier", type=str, default="none")
  parser.add_argument("--regressor", type=str, default="perceptron:1")
  return parser.parse_args()

def create_network(opt):
  """Creates a neural network according to the given options.

  Args:
    opt : options for neural network creation.
  """
  net_split = opt.net.split(":")
  if net_split[0] == "compact":
    network = Compact(*map(int, net_split[1:]))
  else:
    raise RuntimeError("Not implemented!")

  regressor_split = opt.regressor.split(":")
  if regressor_split[0] == "perceptron":
    regressor = Perceptron(
      2 * int(net_split[3]), int(regressor_split[1])
    )
  elif regressor_split[0] == "mlp":
    regressor = MLP(
      int(net_split[4]), int(net_split[1]), *map(int, regressor_split[1:])
    )
  elif regressor_split[0] == "none":
    regressor = lambda x: x

  result = Siamese(network, regressor)

  return result

def net_name(opt):
  """Generates a unique identifier for a given set of training options.

  Args:
    opt : neural network training options.
  """
  return f"{opt.net}-{opt.classifier}-{opt.regressor}-{opt.task}"

def show_images(training, inputs, labels):
  """Callback to display validation images during training.
  
  Args:
    training (Training): neural network training algorithm.
    inputs (torch.Tensor): batch inputs.
    labels (torch.Tensor): batch labels.
  """
  img = inputs[0]
  img = img - img.min()
  img = img / img.max()
  label = labels[0][0]
  training.writer.add_image(f"valid image {label}", img, training.step_id)
  del img

def train(net, opt, data):
  """Trains a given neural network according to the given options."""
  print("Start training ...")
  print("Loading data ...")
  if opt.task == 0:
    data.focus_only = True
  valid_data = copy(data)
  valid_data.data_mode = DataMode.VALID
  valid_data.transform = Normalize()
  print("Done loading data.")
  print("Setting up objectives ...")
  mse = nn.MSELoss()
  losses = [mse]
  print("Done setting up objectives.")
  print("Starting optimization ...")
  training = SupervisedTraining(
    net,
    data,
    valid_data,
    losses,
    batch_size=32,
    network_name=net_name(opt),
    device="cuda:0",
    max_epochs=500,
    valid_callback=show_images
  )
  return training.train()

if __name__ == "__main__":
  print("Parsing arguments ...")
  opt = parse_args()
  print("Arguments parsed.")
  print("Creating network ...")
  net = create_network(opt)
  print("Network created.")
  print("Loading data ...")
  data = BrightfieldFocusDistance(transform=Compose([
    Normalize(),
    Rotation4(),
    Zoom((0.5, 1.5), fill_mode="constant"),
    Translation((0.2, 0.2)),
    Perturb(std=0.5),
    Shift(shift=(0.1, 0.9), scale=(0.1, 0.9)),
    Normalize()
  ]), data_mode=DataMode.TRAIN)
  train(net, opt, data)
  netwrite(net, net_name(opt) + f"-network-final.torch")
