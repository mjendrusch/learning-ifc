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
    
    # self.drop = nn.Dropout2d(p=0.25)

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

class Metric(nn.Module):
  def __init__(self):
    super(Metric, self).__init__()
    self.linear = nn.Linear(512, 1)

  def forward(self, input, prototype):
    # print(input.size())
    # cat = torch.cat((input, prototype.expand(input.size(0), *prototype.size()[1:])), dim=1)
    # return self.linear(cat.view(cat.size(0), -1))
    return torch.norm((input - prototype), p=2, dim=1, keepdim=True)

class Prototype(nn.Module):
  def __init__(self, embedding, metric, way=3):
    super(Prototype, self).__init__()
    self.embedding = embedding
    self.metric = metric
    self.way = way

  def forward(self, data, support, support_labels):
    # print(support.size(), support_labels.size())
    support_embedded = self.embedding(support)
    # print("SE", support_embedded.size())
    shot = support_embedded.size(0) // self.way
    prototype = [
      None
      for label in range(self.way)
    ]
    for label in range(self.way):
      index = int(support_labels.squeeze()[label * shot])
      value = support_embedded[label*shot:(label+1)*shot].mean(dim=0, keepdim=True)
      # print("VALUE", value.size())
      prototype[index] = value
    # print(prototype)
    # print(support_labels.squeeze())
    prototype = torch.cat(prototype, dim=0)
    # print("PS", prototype[:, :2])
    data_embedded = self.embedding(data)
    distances = [
      -self.metric(data_embedded, prototype[label:label+1])
      for label in range(self.way)
    ]
    distances = torch.cat(distances, dim=1).squeeze()
    # print("DS", distances.size())
    return distances

class Refocus(nn.Module):
  def __init__(self, depth, channels,
               filters=2, activation=func.relu):
    super(Refocus, self).__init__()
    self.preprocessor = nn.Conv2d(channels, 2 ** filters, 3, padding=1)
    self.blocks = nn.ModuleList([
      nn.Conv2d(2 ** (filters + idx), 2 ** (filters + idx + 1), 3,
                padding=idx + 1, dilation=idx + 1)
      for idx in range(depth)
    ])
    self.bn_p = nn.BatchNorm2d(2 ** filters)
    self.bn = nn.ModuleList([
      nn.BatchNorm2d(2 ** (filters + idx + 1))
      for idx in range(depth)
    ])
    self.postprocessor = nn.Conv2d(2 ** (filters + depth), channels, 1)
    self.activation = activation

  def forward(self, x):
    out = self.activation(self.preprocessor(x))
    out = self.bn_p(out)
    for bn, block in zip(self.bn, self.blocks):
      out = self.activation(block(out))
      out = bn(out)
    out = self.postprocessor(out)
    return out

class UnsupervisedEncoder(nn.Module):
  def __init__(self, depth, in_channels, out_channels, category=None,
               filters=2, activation=func.relu, batch_norm=True):
    super(UnsupervisedEncoder, self).__init__()

    self.category = category
    if self.category is not None:
      self.categorical = nn.Conv2d(2 ** (filters + depth), category, 1)
    self.activation = activation
    self.preprocessor = nn.Conv2d(in_channels, 2 ** filters, 3, padding=1)
    self.bn_p = nn.BatchNorm2d(2 ** filters)
    self.blocks = nn.ModuleList([
      CompactModule(2 ** (filters + idx), 2 ** (filters + idx + 1))
      for idx in range(depth)
    ])
    self.bn = nn.ModuleList([
      nn.BatchNorm2d(2 ** (filters + idx + 1))
      for idx in range(depth)
    ])
    self.mean = nn.Conv2d(2 ** (filters + depth), out_channels, 1)
    self.std = nn.Conv2d(2 ** (filters + depth), out_channels, 1)
    self.postprocessor = nn.Conv2d(2 ** (filters + depth), out_channels, 1)

  def forward(self, x):
    out = self.bn_p(self.activation(self.preprocessor(x)))
    for idx, module in enumerate(self.blocks):
      bn = self.bn[idx]
      out = bn(self.activation(module(out)))
    mean = self.mean(out)
    std = self.std(out)
    features = func.adaptive_avg_pool2d(self.postprocessor(out), 1)
    if self.category is not None:
      logits = func.softmax(
        func.adaptive_max_pool2d(
          self.categorical(out), 1
        ), dim=1
      )
      return features, mean, std, logits
    return features, mean, std

class UnsupervisedDecoder(nn.Module):
  def __init__(self, depth, in_channels, out_channels, category=None,
               filters=2, activation=func.relu, batch_norm=True):
    super(UnsupervisedDecoder, self).__init__()

    self.category = category
    if self.category is not None:
      self.categorical = nn.Linear(category, out_channels)

    self.activation = activation
    self.preprocessor = nn.Conv2d(2 ** filters, in_channels, 3, padding=1)
    self.bn_p = nn.BatchNorm2d(in_channels)
    self.blocks = nn.ModuleList([
      CompactModule(2 ** (filters + idx + 1), 2 ** (filters + idx), do_pool=False)
      for idx in reversed(range(depth))
    ])
    self.bn = nn.ModuleList([
      nn.BatchNorm2d(2 ** (filters + idx))
      for idx in reversed(range(depth + 1))
    ])
    self.postprocessor = nn.Conv2d(out_channels, 2 ** (filters + depth), 1)

  def forward(self, x, categorical=None):
    if categorical is not None:
      # print(categorical.size(), x.size())
      cat_part = self.categorical(categorical.view(
        categorical.size(0), -1
      )).unsqueeze(2).unsqueeze(2)
      x = self.activation(x + cat_part)
    out = self.activation(self.postprocessor(x))
    for idx, module in enumerate(self.blocks):
      bn = self.bn[idx]
      out = bn(out)
      out = func.interpolate(out, scale_factor=2, mode="bilinear")
      out = self.activation(module(out))
    out = self.bn[-1](out)
    out = self.preprocessor(out)
    out = self.bn_p(out)
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

class Siamese(nn.Module):
  def __init__(self, net, task):
    super(Siamese, self).__init__()
    self.net = net
    self.task = task

  def forward(self, input):
    # values = self.net(input.view(
    #   input.size(0) * input.size(1),
    #   1,
    #   input.size(2),
    #   input.size(3)
    # ))
    # values = values.reshape(*input.size())
    left, right = self.net(input[:, 0:-1]), self.net(input[:, -1:])
    # for idx in range(input.size(1) - 1):
      # task_result = self.task(torch.cat((values[:, idx:idx+1], values[:, -1:]), dim=1))
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
      self.categorical = nn.Linear((128 // (2 ** (depth + 1))) ** 2 * 2 ** (filters + depth), self.category)

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
               filters=2, activation=func.relu, batch_norm=True):
    super(CompactDecoder, self).__init__()

    self.category = category
    if self.category is not None:
      self.categorical = nn.Linear(category, out_channels)

    self.activation = activation
    self.preprocessor = nn.ConvTranspose2d(2 ** filters, in_channels, 3, stride=2, padding=1, output_padding=1)
    self.bn_p = nn.BatchNorm2d(in_channels)
    self.blocks = nn.ModuleList([
      nn.ConvTranspose2d(2 ** (filters + idx + 1), 2 ** (filters + idx), 3, stride=2, padding=1, output_padding=1)
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
    # print(categorical.size(), x.size())
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

class CompactAE(nn.Module):
  def __init__(self, depth, in_channels, out_channels,
               filters=2, activation=func.relu, batch_norm=True):
    super(CompactAE, self).__init__()

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
    self.postprocessor = nn.Conv2d(2 ** (filters + depth), out_channels, 1)

  def forward(self, x):
    out = self.activation(self.preprocessor(x))
    for idx, module in enumerate(self.blocks):
      bn = self.bn[idx]
      out = bn(self.activation(module(out)))
    return self.postprocessor(out)

class CompactAD(nn.Module):
  def __init__(self, depth, in_channels, out_channels,
               filters=2, activation=func.relu, batch_norm=True):
    super(CompactAD, self).__init__()

    self.activation = activation
    self.preprocessor = nn.ConvTranspose2d(2 ** filters, in_channels, 3, stride=2, padding=1, output_padding=1)
    self.bn_p = nn.BatchNorm2d(in_channels)
    self.blocks = nn.ModuleList([
      nn.ConvTranspose2d(2 ** (filters + idx + 1), 2 ** (filters + idx), 3, stride=2, padding=1, output_padding=1)
      for idx in reversed(range(depth))
    ])
    self.bn = nn.ModuleList([
      nn.BatchNorm2d(2 ** (filters + idx))
      for idx in reversed(range(depth + 1))
    ])
    self.postprocessor = nn.ConvTranspose2d(out_channels, 2 ** (filters + depth), 3, stride=1, padding=1)

  def forward(self, x):
    out = self.activation(self.postprocessor(x))
    for idx, module in enumerate(self.blocks):
      bn = self.bn[idx]
      out = bn(out)
      out = self.activation(module(out))
    out = self.bn[-1](out)
    out = self.preprocessor(out)
    out = self.bn_p(out)
    return out