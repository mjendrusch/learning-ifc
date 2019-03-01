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
# from torchsupport.training.clustering import ClusteringTraining, HierarchicalClusteringTraining, DEPICTTraining, ClusterAETraining
from torchsupport.training.vae import JointVAETraining

from learning_ifc.learning.models.compact import CompactAE, CompactAD, Compact, DenseCompact, Perceptron, MLP, Multitask, UnsupervisedEncoder, UnsupervisedDecoder, CompactEncoder, CompactDecoder
from learning_ifc.datasets.brightfield import BrightfieldDeviceImage, BrightfieldImage
from matplotlib import pyplot as plt

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
  parser = ArgumentParser(description="Unsupervisied classification training.")
  parser.add_argument("--epochs", type=int, default=5000)
  parser.add_argument("--category", type=int, default=10)
  parser.add_argument("--continuous", type=int, default=64)
  parser.add_argument("--gamma", type=float, default=1000.0)
  parser.add_argument("--ctarget", type=float, default=50)
  parser.add_argument("--dtarget", type=float, default=5)
  parser.add_argument("--net", type=str, default="compact:6:1:256:2")
  return parser.parse_args()

def create_network(opt):
  net_split = opt.net.split(":")
  if net_split[0] == "compact":
    network = Compact(*map(int, net_split[1:]))
  else:
    raise RuntimeError("Not implemented!")

  network = (
    # Compact(5, 1, 256, filters=2)
    # CompactAE(5, 1, 256, filters=4),
    # CompactAD(5, 1, 256, filters=4)
    CompactEncoder(5, 1, opt.continuous, filters=4, category=opt.category),
    CompactDecoder(5, 1, opt.continuous, filters=4, category=opt.category)
  )
  return network

def net_name(opt):
  return f"{opt.net}"

def train(net, opt, data):
  print("Start training ...")
  print("Setting up objectives ...")
  print("Done setting up objectives.")
  print("Starting optimization ...")
  # training = DEPICTTraining(
  #   net[0], net[1], nn.Linear(256, 10),
  #   data,
  #   clustering=KMeans(10),
  #   batch_size=64,
  #   loss=nn.CrossEntropyLoss(),
  #   network_name=net_name(opt) + "-simple-clustering",
  #   device="cuda:0",
  #   max_epochs=500
  # )
  training = JointVAETraining(
    net[0], net[1],
    data,
    gamma=opt.gamma,
    ctarget=opt.ctarget,
    dtarget=opt.dtarget,
    batch_size=64,
    network_name=net_name(opt) + f"-VAE-joint-{opt.category}-{opt.continuous}-{opt.gamma}-{opt.dtarget}-{opt.ctarget}",
    device="cuda:0",
    max_epochs=opt.epochs
  )
  # training = ClusteringTraining(
  #   net,
  #   data,
  #   clustering=KMeans(3),
  #   # depth=[3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
  #   # loss=ce,
  #   batch_size=64,
  #   network_name=net_name(opt) + "-orderless-clustering-3",
  #   device="cuda:0",
  #   max_epochs=500
  # )
  # training = ClusterAETraining(
  #   net[0], net[1],
  #   data,
  #   # depth=[3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
  #   # loss=ce,
  #   batch_size=64,
  #   network_name=net_name(opt) + "-hardened-clustering",
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
    Affine(
      rotation_range=360,
      zoom_range=(0.9, 1.1),
      fill_mode="reflect"
    ),
    Affine(
      translation_range=(0.5, 0.5),
      fill_mode="constant",
      fill_value=0.5
    ),
    Perturb(std=0.1),
    MinMax()
  ]))
  train(net, opt, data)
  netwrite(net, net_name(opt) + f"-network-final.torch")
