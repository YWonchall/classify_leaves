import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

class Accumulator: 
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

def show_images(imgs, num_rows=2, num_cols=5, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def try_gpu(i=0):
    assert torch.cuda.device_count() >= i + 1
    return torch.device(f"cuda:{i}")

def to_devices(net, device, num_gpus=1):
    if device == 'gpu':
        devices = [try_gpu(i) for i in range(num_gpus)]
        net.to(devices[0])
        if num_gpus>1:
            net = nn.DataParallel(net, device_ids=devices)
    else:
        devices = ['cpu']
    return net, devices

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def evaluate_accuracy(net, data_iter, device, num_gpus=1):
    net, devices = to_devices(net, device, num_gpus)
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(devices[0]), y.to(devices[0])
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
