"""Supervised single-task classification training for classifying
yeast species from brightfield images."""
from argparse import ArgumentParser
from copy import copy

import torch
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torchsupport.data.io import netwrite
from torchsupport.data.transforms import Compose, Perturb, Normalize, MinMax, Affine
from torchsupport.training.vae import FactorVAETraining

from learning_ifc.learning.models.compact import MLP, CompactEncoder, CompactDecoder
from learning_ifc.datasets.brightfield import BrightfieldDeviceImage


def parse_args():
  """Parses neural network training arguments."""
  parser = ArgumentParser(description="Unsupervisied classification training.")
  parser.add_argument("--epochs", type=int, default=5000)
  parser.add_argument("--category", type=int, default=10)
  parser.add_argument("--continuous", type=int, default=64)
  parser.add_argument("--gamma", type=float, default=1000.0)
  parser.add_argument("--ctarget", type=float, default=50)
  parser.add_argument("--dtarget", type=float, default=5)
  parser.add_argument("--net", type=str, default="factor")
  return parser.parse_args()

def create_network(opt):
  """Creates a neural network according to the given options.

  Args:
    opt : options for neural network creation.
  """
  network = (
    MLP(opt.continuous, 2, 2, hidden=64),
    CompactEncoder(5, 1, opt.continuous, filters=4, category=None),
    CompactDecoder(5, 1, opt.continuous, filters=4, category=None)
  )
  return network

def net_name(opt):
  """Generates a unique identifier for a given set of training options.

  Args:
    opt : neural network training options.
  """
  return f"{opt.net}" + \
    f"-VAE-{opt.category}-{opt.continuous}-{opt.gamma}-{opt.dtarget}-{opt.ctarget}"

def train(net, opt, data):
  """Trains a given neural network according to the given options."""
  print("Start training ...")
  print("Setting up objectives ...")
  print("Done setting up objectives.")
  print("Starting optimization ...")
  training = FactorVAETraining(
    net[1], net[2], net[0],
    data,
    gamma=opt.gamma,
    ctarget=opt.ctarget,
    dtarget=opt.dtarget,
    batch_size=256,
    network_name=net_name(opt),
    device="cuda:0",
    max_epochs=opt.epochs
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
  data = BrightfieldDeviceImage(transform=Compose([
    Normalize(),
    Affine(
      rotation_range=360,
      zoom_range=(0.9, 1.1),
      fill_mode="reflect"
    ),
    Perturb(std=0.1),
    MinMax()
  ]))
  train(net, opt, data)
  netwrite(net[1], net_name(opt) + f"-encoder-final.torch")
  netwrite(net[2], net_name(opt) + f"-decoder-final.torch")
  netwrite(net[0], net_name(opt) + f"-discriminator-final.torch")
