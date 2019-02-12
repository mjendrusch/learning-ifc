"""Supervised single-task classification training for classifying
yeast species from brightfield images."""
from argparse import ArgumentParser
from copy import copy

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from tensorboardX import SummaryWriter

from torchsupport.data.io import netread, netwrite
from torchsupport.data.transforms import Rotation4, Elastic, Compose, Perturb, Normalize, Affine
from torchsupport.training.training import SupervisedTraining

from learning_ifc.learning.models.compact import Compact, DenseCompact, Perceptron, MLP, Multitask
from learning_ifc.datasets.brightfield import Brightfield, DataMode

from matplotlib import pyplot as plt

def parse_args():
  parser = ArgumentParser(description="Supervisied classification training.")
  parser.add_argument("--test", dest="test", action="store_true")
  parser.set_defaults(test=False)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--task", type=int, default=2)
  parser.add_argument("--net", type=str, default="compact:6:1:256:2")
  parser.add_argument("--classifier", type=str, default="perceptron:3")
  parser.add_argument("--regressor", type=str, default="perceptron:1")
  return parser.parse_args()

def create_network(opt):
  net_split = opt.net.split(":")
  if net_split[0] == "compact":
    network = Compact(*map(int, net_split[1:]))
  else:
    raise RuntimeError("Not implemented!")

  classifier_split = opt.classifier.split(":")
  if classifier_split[0] == "perceptron":
    classifier = Perceptron(
      int(net_split[4]), int(net_split[1]), int(classifier_split[1])
    )
  elif classifier_split[0] == "mlp":
    classifier = MLP(
      int(net_split[4]), int(net_split[1]), *map(int, classifier_split[1:])
    )
  elif classifier_split[0] == "none":
    classifier = lambda x: x

  regressor_split = opt.regressor.split(":")
  if regressor_split[0] == "perceptron":
    regressor = Perceptron(
      int(net_split[4]), int(net_split[1]), int(regressor_split[1])
    )
  elif regressor_split[0] == "mlp":
    regressor = MLP(
      int(net_split[4]), int(net_split[1]), *map(int, regressor_split[1:])
    )
  elif regressor_split[0] == "none":
    regressor = lambda x: x

  if opt.task == 0:
    result = Multitask(network, [regressor])
  elif opt.task == 1:
    result = Multitask(network, [classifier])
  else:
    result = Multitask(network, [classifier, regressor])

  return result

def net_name(opt):
  return f"{opt.net}-{opt.classifier}-{opt.regressor}-{opt.task}"

def show_images(training, inputs, labels):
  img = inputs[0]
  img = img - img.min()
  img = img / img.max()
  training.writer.add_image(f"example image", img, training.step_id)
  del img

def train(net, opt, data):
  print("Start training ...")
  print("Loading data ...")
  if opt.task == 0:
    data.focus_only = True
  valid_data = copy(data)
  valid_data.transform = Normalize()
  valid_data.data_mode = DataMode.VALID
  print("Done loading data.")
  print("Setting up objectives ...")
  mse = nn.MSELoss()
  ce = nn.CrossEntropyLoss()
  if opt.task == 0:
    losses = [
      mse
    ]
  elif opt.task == 1:
    losses = [
      ce
    ]
  else:
    losses = [
      ce,
      mse
    ]
  print("Done setting up objectives.")
  print("Starting optimization ...")
  training = SupervisedTraining(
    net,
    data,
    valid_data,
    losses,
    network_name=net_name(opt),
    device="cuda:0",
    max_epochs=20,
    valid_callback=show_images
  )
  return training.train()

def test(net, opt, data):
  with torch.no_grad():
    classification = np.zeros((len(data), 4))
    rmse = np.zeros(len(data))
    if opt.task == 0:
      data.data_mode = DataMode.TRAIN
      data.focus_only = True
      for idx, (elem, focus) in enumerate(data):
        focus_p = net(elem.unsqueeze(0))[0]
        rmse[idx] = (focus_p - focus).mean()
        print(float(focus_p), float(focus))
    elif opt.task == 1:
      for idx, (elem, label) in enumerate(data):
        classification[idx, :3] = net(elem.unsqueeze(0)).numpy()
        classification[3] = label
    elif opt.task == 2:
      for idx, (elem, label, focus) in enumerate(data):
        logits, focus_p = net(elem.unsqueeze(0))
        if int(label) != 0:
          print(int(logits.argmax()), float(focus_p), int(label), float(focus))
          if int(logits.argmax()) != int(label):
            plt.imshow(elem.numpy()[0]); plt.show()
        classification[idx, :3] = logits.numpy()
        classification[3] = label
        rmse[idx] = (focus_p - focus).mean()
    return classification, rmse

if __name__ == "__main__":
  print("Parsing arguments ...")
  opt = parse_args()
  print("Arguments parsed.")
  print("Creating network ...")
  net = create_network(opt)
  print("Network created.")
  print("Loading data ...")
  data = Brightfield(transform=Compose([
    Normalize(),
    Rotation4(),
    Affine(
      translation_range=(5, 5),
      zoom_range=(0.9, 1.1),
      fill_mode="reflect"
    ),
    Perturb(std=(0.0, 0.1))
  ]), data_mode=DataMode.TRAIN)
  if opt.test:
    netread(net, net_name(opt) + f"-network-final.torch")
    data.data_mode = DataMode.TEST
    data.transform = Compose([
      Normalize(),
      Affine(
        rotation_range=360,
        translation_range=(50, 50),
        zoom_range=(0.8, 1.2),
        fill_mode="reflect"
      )
    ])
    classification, rmse = test(net, opt, data)
    np.save(net_name(opt) + "-rmse.npy", rmse)
    np.save(net_name(opt) + "-class.npy", classification)
  else:
    train(net, opt, data)
    netwrite(net, net_name(opt) + f"-network-final.torch")
