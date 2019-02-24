"""Supervised single-task classification training for classifying
yeast species from brightfield images."""
from argparse import ArgumentParser
from copy import copy

from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch import optim

from tensorboardX import SummaryWriter

from torchsupport.data.io import netread, netwrite
from torchsupport.data.transforms import Rotation4, Elastic, Compose, Shift, Zoom, Perturb, Normalize, MinMax, Center, Affine
from torchsupport.training.training import SupervisedTraining

from learning_ifc.learning.models.compact import Refocus
from learning_ifc.datasets.brightfield import BrightfieldRefocus
from matplotlib import pyplot as plt

def parse_args():
  parser = ArgumentParser(description="Defocusing training.")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--net", type=str, default="refocus:3:1:2")
  return parser.parse_args()

def create_network(opt):
  net_split = opt.net.split(":")
  if net_split[0] == "refocus":
    network = Refocus(*map(int, net_split[1:]))
  else:
    raise RuntimeError("Not implemented!")
  return network

def net_name(opt):
  return f"{opt.net}"

def train(net, opt, data):
  print("Start training ...")
  print("Setting up objectives ...")
  print("Done setting up objectives.")
  print("Starting optimization ...")
  training = SupervisedTraining(
    net,
    data,
    data,
    losses=[nn.MSELoss()],
    batch_size=64,
    network_name=net_name(opt) + "-defocus",
    device="cuda:0",
    max_epochs=500,
    valid_callback=lambda x, y, z: x
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
  data = BrightfieldRefocus(transform=Compose([
    Normalize(),
    Affine(
      rotation_range=360,
      translation_range=(0.5, 0.5),
      zoom_range=(0.9, 1.1),
      fill_mode="reflect"
    ),
    Perturb(mean=(-0.5, 0.5), std=(0.2, 0.5)),
    Normalize()
  ]))
  train(net, opt, data)
  netwrite(net, net_name(opt) + f"-network-final.torch")
