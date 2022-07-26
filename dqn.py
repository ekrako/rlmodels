# from here -> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# %%
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# %%


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def create_conv_block(in_channels, out_channels, kernel_size,
                      stride=1, padding=0, pool_kernel_size=2, pool_stride=1,
                      relu_inplace=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=relu_inplace),
        nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),

    )


class DQN(nn.Module):

    def __init__(self, h=128, w=128, outputs=5, in_channels=4, device='cpu',
                 kernel_size=3, conv_channels=32, stride = 1,
                 hidden_nodes1=512, hidden_nodes2=256, hidden_nodes3=64):
        super(DQN, self).__init__()
        self.device = device
        self.cnn = nn.Sequential(
            # 2D convolutional layer
            create_conv_block(in_channels, conv_channels, kernel_size),
            create_conv_block(conv_channels, 4, kernel_size, padding=1)
        )

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(w, kernel_size, stride),kernel_size, stride)
        convh = conv2d_size_out(conv2d_size_out(h, kernel_size, stride),kernel_size, stride)
        linear_input_size = convw * convh * 4

        self.head = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(self.linear_input_size, outputs)
            nn.Linear(linear_input_size, hidden_nodes1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes1, hidden_nodes2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes2, hidden_nodes3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes3, outputs)
        )
        self.to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.float().to(self.device)
        if x.ndim == 3:
            x = torch.unsqueeze(x, 0)
        cnn_out = self.cnn(x)
        logits = self.head(cnn_out)
        return logits