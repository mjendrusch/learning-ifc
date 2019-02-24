"""Supervised single-task classification training for classifying
yeast species from brightfield images."""
from argparse import ArgumentParser
from copy import copy

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch import optim

from tensorboardX import SummaryWriter

from torchsupport.data.io import netread, netwrite
from torchsupport.data.transforms import Rotation4, Elastic, Compose, Shift, Zoom, Perturb, PerturbUniform, MinMax, Normalize, Affine, Translation
from torchsupport.training.training import SupervisedTraining

from learning_ifc.learning.models.compact import Compact, DenseCompact, Perceptron, MLP, Multitask, Siamese
from learning_ifc.datasets.brightfield import Brightfield, BrightfieldClass, BrightfieldDevice, BrightfieldFocusDistance, BrightfieldDeviceValid, DataMode

from matplotlib import pyplot as plt

def parse_args():
  parser = ArgumentParser(description="Supervisied classification training.")
  parser.add_argument("--test", dest="test", action="store_true")
  parser.set_defaults(test=False)
  parser.add_argument("--testdevice", type=str, default=None)
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
      int(net_split[3]), int(classifier_split[1])
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
  return f"{opt.net}-{opt.classifier}-{opt.regressor}-{opt.task}"

def show_images(training, inputs, labels):
  img = inputs[0]
  img = img - img.min()
  img = img / img.max()
  label = labels[0][0]
  training.writer.add_image(f"valid image {label}", img, training.step_id)
  del img

def train(net, opt, data):
  print("Start training ...")
  print("Loading data ...")
  if opt.task == 0:
    data.focus_only = True
  valid_data = copy(data)
  valid_data.data_mode = DataMode.VALID
  valid_data.transform = Normalize()#Compose([
  #   MinMax(),
  #   Normalize(),
  #   Affine(
  #     rotation_range=360,
  #     translation_range=(0.2, 0.2),
  #     fill_mode="reflect"
  #   ),
  #   Zoom((0.5, 1.5), fill_mode="reflect"),
  #   Perturb(std=0.5),
  #   Shift(shift=(0.1, 0.9), scale=(0.1, 0.9)),
  #   MinMax(),
  #   Normalize()
  # ])
  print("Done loading data.")
  print("Setting up objectives ...")
  mse = nn.MSELoss()

  if opt.task == 0:
    losses = [
      mse
    ]
  elif opt.task == 1:
    losses = [
      mse
    ]
  else:
    losses = [
      mse
    ]
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
    elif opt.task == 1:
      for idx, (elem, label) in enumerate(data):
        classification[idx, :3] = net(elem.unsqueeze(0)).numpy()
        classification[3] = label
    elif opt.task == 2:
      confuse = np.zeros((3, 41, 3, 41))
      for idx, (elem, label, focus) in enumerate(data):
        logits, focus_p = net(elem.unsqueeze(0))
        predicted_strain = logits.argmax()
        predicted_focus = focus_p.argmax()
        confuse[
          int(label), int(focus),
          int(predicted_strain), int(predicted_focus)
        ] += 1
        # if int(label) != 0:
        #   print(int(logits.argmax()), int(focus_p.argmax()), int(label), int(focus))
        #   if int(logits.argmax()) != int(label):
        #     plt.imshow(elem.numpy()[0]); plt.show()
        classification[idx, :3] = logits.numpy()
        classification[3] = label
        rmse[idx] = (focus_p - focus).mean()
      np.save("confuse.npy", confuse)
    return classification, rmse

def testdevice(net, cer=0.4, lud=0.3, pom=0.3):
  classes = ["cerevisiae", "ludwigii", "pombe"]
  with torch.no_grad():
    net.eval()
    data = BrightfieldDevice([cer, lud, pom], transform=Normalize(), seed=np.random.randint(0, 1254378))
    typ_c = np.zeros(3, dtype=np.int)
    focus_c = np.zeros(41, dtype=np.int)
    for item, typ in data:
      typ_p, *_ = net(item.unsqueeze(0))
      print(classes[int(typ_p.argmax())])#, (int(focus_p.argmax()) - 20) * 0.25)
      # focus_c[int(focus_p.argmax())] += 1
      typ_c[int(typ_p.argmax())] += 1
      plt.imshow(item[0].numpy()); plt.show()
    np.save(f"simrun-{cer}-{lud}-{pom}-focus.npy", focus_c)
    np.save(f"simrun-{cer}-{lud}-{pom}-strain.npy", typ_c)

if __name__ == "__main__":
  print("Parsing arguments ...")
  opt = parse_args()
  print("Arguments parsed.")
  print("Creating network ...")
  net = create_network(opt)
  print("Network created.")
  print("Loading data ...")
  if opt.testdevice is not None:
    netread(net, net_name(opt) + f"-network-final.torch")
    cer, lud, pom = map(float, opt.testdevice.split(":"))
    testdevice(net, cer=cer, lud=lud, pom=pom)
  else:
    data = BrightfieldFocusDistance(transform=Compose([
      Normalize(),
      Rotation4(),
      Zoom((0.5, 1.5), fill_mode="constant"),
      Translation((0.2, 0.2)),
      Perturb(std=0.5),
      Shift(shift=(0.1, 0.9), scale=(0.1, 0.9)),
      Normalize()
    ]), data_mode=DataMode.TRAIN)
    if opt.test:
      netread(net, net_name(opt) + f"-network-final.torch")
      net.eval()
      data.data_mode = DataMode.TEST
      data.transform = Compose([
        Normalize()
      ])
      classification, rmse = test(net, opt, data)
      np.save(net_name(opt) + "-rmse.npy", rmse)
      np.save(net_name(opt) + "-class.npy", classification)
    else:
      train(net, opt, data)
      netwrite(net, net_name(opt) + f"-network-final.torch")
