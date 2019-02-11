import torch.nn as nn
import torch.nn.functional as func
from torchsupport.modules.separable import DepthWiseSeparableConv2d

class CompactModule(nn.Module):
  def __init__(self, in_channels, out_channels,
               activation=func.relu, batch_norm=True):
    super(CompactModule, self).__init__()
    self.activation = activation
    self.layers = nn.ModuleList([
      DepthWiseSeparableConv2d(in_channels, out_channels, 3, padding=1),
      nn.Conv2d(out_channels, in_channels, 1, padding=1),
      DepthWiseSeparableConv2d(in_channels, out_channels, 3, padding=1)
    ])
    self.pool = nn.MaxPool2d(2)

    self.batch_norm = batch_norm
    if self.batch_norm:
      self.bn = nn.ModuleList([
        nn.BatchNorm2d(out_channels),
        nn.BatchNorm2d(in_channels),
        nn.BatchNorm2d(out_channels)
      ])

  def forward(self, x):
    out = x
    for idx, module in enumerate(self.layers):
      out = module(out)
      out = self.bn[idx](out) if self.batch_norm else out
      out = self.activation(out)
    return self.pool(out)

class FullCompactModule(CompactModule):
  def __init__(self, in_channels, out_channels,
               activation=func.relu, batch_norm=True):
    super(FullCompactModule, self).__init__(
      in_channels, out_channels,
      activation=activation, batch_norm=batch_norm
    )
    self.layers[0] = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    self.layers[2] = nn.Conv2d(in_channels, out_channels, 3, padding=1)

class Compact(nn.Module):
  def __init__(self, depth, in_channels, out_channels,
               filters=4, activation=func.relu, batch_norm=True):
    super(Compact, self).__init__()
    self.activation = activation
    self.preprocessor = FullCompactModule(
      in_channels, 2 ** filters,
      activation=activation,
      batch_norm=batch_norm
    )
    self.postprocessor = nn.Conv2d(2 ** (filters + depth), out_channels, 1)
    self.batch_norm = batch_norm
    self.layers = nn.ModuleList([
      CompactModule(
        2 ** (filters + idx),
        2 ** (filters + idx + 1),
        activation=activation,
        batch_norm=batch_norm
      )
      for idx in range(depth)
    ])

  def forward(self, x):
    out = self.preprocessor(x)
    for module in self.layers:
      out = module(out)
    out = self.postprocessor(out)
    out = func.adaptive_avg_pool2d(out, 1)
    return out

class DenseCompact(Compact):
  def __init__(self, depth, in_channels, out_channels,
               filters=4, activation=func.relu, batch_norm=True):
    super(DenseCompact, self).__init__(
      depth, in_channels, out_channels, filters=filters,
      activation=activation, batch_norm=batch_norm
    )
    self.translators = nn.ModuleList([
      nn.Conv2d(2 ** (filters + idx), 2 ** (filters + idx + 1), 1)
      for idx in range(depth)
    ])
    self.pool = nn.MaxPool2d(2)

  def forward(self, x):
    out = self.preprocessor(x)
    inputs = [out]
    for idx, module in enumerate(self.layers):
      inp = sum(inputs)
      out = module(inp)
      new_inputs = [
        self.pool(self.translators[idx](inp))
        for inp in inputs
      ]
      new_inputs.append(out)
      inputs = new_inputs
    out = self.postprocessor(out)
    return self.adaptive_avg_pool2d(out, 1)

class Perceptron(nn.Module):
  def __init__(self, filters, depth, classes,
               final_activation=lambda x: x):
    super(Perceptron, self).__init__()
    self.final_activation = final_activation
    self.linear = nn.Linear(2 ** (filters + depth), classes)

  def forward(self, input):
    return self.final_activation(self.linear(input.squeeze()))

class MLP(nn.Module):
  def __init__(self, filters, depth, classes, layers,
               hidden=128, final_activation=lambda x: x):
    super(MLP, self).__init__()
    self.final_activation = final_activation
    self.preprocessor = nn.Linear(2 ** (filters + depth), hidden)
    self.linear = nn.ModuleList([
      nn.Linear(
        hidden,
        hidden
      )
      for _ in range(layers - 2)
    ])
    self.postprocessor = nn.Linear(hidden, classes)

  def forward(self, input):
    out = func.relu(self.preprocessor(input.squeeze()))
    for module in self.linear:
      out = func.relu(module(out))
    out = self.final_activation(self.postprocessor(out))
    return out

class Multitask(nn.Module):
  def __init__(self, net, tasks):
    super(Multitask, self).__init__()
    self.net = net
    self.tasks = nn.ModuleList(tasks)

  def forward(self, input):
    preprocessed = self.net(input)
    outputs = []
    for task in self.tasks:
      outputs.append(task(preprocessed))
    return outputs
