import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.modules.separable import DepthWiseSeparableConv2d

class CompactModule(nn.Module):
  def __init__(self, in_channels, out_channels,
               activation=func.relu, batch_norm=True,
               residual=True, do_pool=True):
    super(CompactModule, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.activation = activation
    self.layers = nn.ModuleList([
      DepthWiseSeparableConv2d(in_channels, out_channels, 3, padding=1),
      nn.Conv2d(out_channels, in_channels, 1),
      DepthWiseSeparableConv2d(in_channels, out_channels, 3, padding=1)
    ])
    self.residual = residual
    self.pool = nn.MaxPool2d(2)
    self.do_pool = do_pool

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
    if self.in_channels == self.out_channels:
      return torch.cat([self.pool(out), self.pool(x)], dim=1)
    if self.do_pool:
      out = self.pool(out)
    return out

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
        2 ** (filters + idx),
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

class Perceptron(nn.Module):
  def __init__(self, out_channels, classes,
               final_activation=lambda x: x):
    super(Perceptron, self).__init__()
    self.final_activation = final_activation
    self.linear = nn.Linear(out_channels, classes)

  def forward(self, input):
    out = input.squeeze()
    out = self.linear(out)
    return self.final_activation(out)

class MLP(nn.Module):
  def __init__(self, out_channels, classes, layers,
               hidden=128, final_activation=lambda x: x):
    super(MLP, self).__init__()
    self.final_activation = final_activation
    self.preprocessor = nn.Linear(out_channels, hidden)
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
      out = func.leaky_relu(module(out))
    out = self.final_activation(self.postprocessor(out))
    return out

class Siamese(nn.Module):
  def __init__(self, net, task):
    super(Siamese, self).__init__()
    self.net = net
    self.task = task

  def forward(self, input):
    left, right = self.net(input[:, 0:-1]), self.net(input[:, -1:])
    return [self.task(torch.cat((left, right), dim=1))]

class CompactEncoder(nn.Module):
  def __init__(self, depth, in_channels, out_channels, category=None,
               filters=2, activation=func.relu, batch_norm=True):
    super(CompactEncoder, self).__init__()

    self.category = category
    self.activation = activation
    self.preprocessor = nn.Conv2d(in_channels, 2 ** filters, 3, stride=2, padding=1)
    self.bn_p = nn.BatchNorm2d(2 ** filters)
    self.blocks = nn.ModuleList([
      nn.Conv2d(2 ** (filters + idx), 2 ** (filters + idx + 1), 3, stride=2, padding=1)
      for idx in range(depth)
    ])
    self.bn = nn.ModuleList([
      nn.BatchNorm2d(2 ** (filters + idx + 1))
      for idx in range(depth)
    ])
    self.mean = nn.Linear((128 // (2 ** (depth + 1))) ** 2 * 2 ** (filters + depth), out_channels)
    self.std = nn.Linear((128 // (2 ** (depth + 1))) ** 2 * 2 ** (filters + depth), out_channels)
    if self.category is not None:
      self.categorical = nn.Linear(
        (128 // (2 ** (depth + 1))) ** 2 * 2 ** (filters + depth),
        self.category
      )

  def forward(self, x):
    out = self.activation(self.preprocessor(x))
    for idx, module in enumerate(self.blocks):
      bn = self.bn[idx]
      out = bn(self.activation(module(out)))
    out = out.reshape(out.size(0), -1)
    mean = self.mean(out)
    std = self.std(out)
    if self.category is not None:
      logits = func.softmax(
        self.categorical(out), dim=1
      )
      return out, mean, std, logits
    return out, mean, std

class CompactDecoder(nn.Module):
  def __init__(self, depth, in_channels, out_channels, category=None,
               filters=2, activation=func.relu):
    super(CompactDecoder, self).__init__()

    self.category = category
    if self.category is not None:
      self.categorical = nn.Linear(category, out_channels)

    self.activation = activation
    self.preprocessor = nn.ConvTranspose2d(
      2 ** filters, in_channels, 3,
      stride=2, padding=1, output_padding=1
    )
    self.bn_p = nn.BatchNorm2d(in_channels)
    self.blocks = nn.ModuleList([
      nn.ConvTranspose2d(
        2 ** (filters + idx + 1), 2 ** (filters + idx), 3,
        stride=2, padding=1, output_padding=1
      )
      for idx in reversed(range(depth))
    ])
    self.bn = nn.ModuleList([
      nn.BatchNorm2d(2 ** (filters + idx))
      for idx in reversed(range(depth + 1))
    ])
    if self.category is not None:
      self.postprocessor = nn.Linear(out_channels + self.category, 4 * 2 ** (filters + depth))
    else:
      self.postprocessor = nn.Linear(out_channels, 4 * 2 ** (filters + depth))

  def forward(self, x, categorical=None):
    if self.category is not None:
      out = self.activation(self.postprocessor(torch.cat((categorical, x), dim=1)))
    else:
      out = self.activation(self.postprocessor(x))
    out = out.reshape(out.size(0), out.size(1) // 4, 2, 2)
    for idx, module in enumerate(self.blocks):
      bn = self.bn[idx]
      out = bn(out)
      out = self.activation(module(out))
    out = self.bn[-1](out)
    out = self.preprocessor(out)
    out = self.bn_p(out)
    out = torch.clamp(torch.sigmoid(out), 0, 1)
    return out
