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
from torchsupport.training.clustering import ClusteringTraining, HierarchicalClusteringTraining, VAEClusteringTraining
from torchsupport.training.vae import JointVAETraining

from learning_ifc.learning.models.compact import Compact, DenseCompact, Perceptron, MLP, Multitask, UnsupervisedEncoder, UnsupervisedDecoder
from learning_ifc.datasets.brightfield import BrightfieldDeviceImage, BrightfieldImage
from matplotlib import pyplot as plt

def parse_args():
  parser = ArgumentParser(description="Unsupervisied classification training.")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--net", type=str, default="compact:6:1:256:2")
  return parser.parse_args()

def create_network(opt):
  net_split = opt.net.split(":")
  if net_split[0] == "compact":
    network = Compact(*map(int, net_split[1:]))
  else:
    raise RuntimeError("Not implemented!")

  network = (
    UnsupervisedEncoder(5, 1, 256, filters=4, category=3),
    UnsupervisedDecoder(5, 1, 256, filters=4, category=3)
  )
  return network

def net_name(opt):
  return f"{opt.net}"

def train(net, opt, data):
  print("Start training ...")
  print("Setting up objectives ...")
  print("Done setting up objectives.")
  print("Starting optimization ...")
  # training = HierarchicalClusteringTraining(
  #   net,
  #   data,
  #   depth=[5, 10, 30, 50],
  #   # loss=ce,
  #   network_name=net_name(opt),
  #   device="cuda:0",
  #   max_epochs=500
  # )
  training = JointVAETraining(
    net[0], net[1],
    data,
    batch_size=64,
    network_name=net_name(opt) + "-VAE-joint",
    device="cuda:0",
    max_epochs=500
  )
  # training = VAEClusteringTraining(
  #   net[0], net[1],
  #   data,
  #   depth=[3, 6],
  #   # loss=ce,
  #   batch_size=64,
  #   network_name=net_name(opt) + "-VAE",
  #   device="cuda:0",
  #   max_epochs=500
  # )
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
    Rotation4(),
    # Affine(
    #   rotation_range=360,
    #   translation_range=(0.0, 0.0),
    #   zoom_range=(0.9, 1.1),
    #   fill_mode="reflect"
    # ),
    Perturb(std=0.05)
  ]))
  train(net, opt, data)
  netwrite(net, net_name(opt) + f"-network-final.torch")
