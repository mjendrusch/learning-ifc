"""Supervised single-task classification training for classifying
yeast species from brightfield images."""
from argparse import ArgumentParser
from copy import copy

from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from tensorboardX import SummaryWriter

from torchsupport.data.io import netread, netwrite
from torchsupport.data.transforms import Rotation4, Elastic, Compose, Perturb, Normalize
from torchsupport.training.training import SupervisedTraining

from learning_ifc.learning.models.compact import Compact, DenseCompact, Perceptron, MLP, Multitask
from learning_ifc.datasets.brightfield import Brightfield, DataMode

def parse_args():
  parser = ArgumentParser(description="Supervisied classification training.")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--task", type=int, default=2)
  parser.add_argument("--net", type=str, default="compact:4:1:256:4")
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
      int(net_split[4]), int(net_split[1]), *map(int, classifier_split[1])
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
      int(net_split[4]), int(net_split[1]), *map(int, regressor_split[1])
    )
  elif regressor_split[0] == "none":
    regressor = lambda x: x

  result = Multitask(network, [classifier]) if opt.task == 1 else Multitask(network, [classifier, regressor])

  return result

def net_name(opt):
  return f"{opt.net}-{opt.classifier}-{opt.regressor}-{opt.task}"

def train(net, opt):
  data = Brightfield(transform=Compose([
    Normalize(),
    Rotation4(),
    Perturb()
  ]), data_mode=DataMode.TRAIN)
  valid_data = copy(data)
  valid_data.data_mode = DataMode.VALID
  training = SupervisedTraining(
    net, data, valid_data, [
      nn.CrossEntropyLoss(),
      nn.MSELoss()
    ], network_name=net_name(opt),
    device="cuda:0"
  )
  return training.train()

if __name__ == "__main__":
  opt = parse_args()
  net = create_network(opt)
  train(net, opt)
  netwrite(net, net_name(opt) + f"-network-final.torch")
