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

def create_conv_block(in_channels, out_channels, kernel_size,
                      stride=1, padding=0, pool_kernel_size=2, pool_stride=1,
                      relu_inplace=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=relu_inplace),
        nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),

    )


class AC2(nn.Module):
    def __init__(self,  h=128, w=128, outputs=5, in_channels=4, device='cpu',
                 kernel_size=3, conv_channels=32, stride = 1,
                 hidden_nodes1=128, hidden_nodes2=64,device='cpu'):
        super(AC2, self).__init__()

        self.feature_extractor = nn.Sequential(
            # 2D conv layer
            create_conv_block(4, 32, 3),
            create_conv_block(32, 64, 3),
            create_conv_block(64, 4, 3),

            
        )
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size, stride),kernel_size, stride),kernel_size, stride)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size, stride),kernel_size, stride),kernel_size, stride)
        linear_input_size = convw * convh * 4

        self.common = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, hidden_nodes1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes1, hidden_nodes2),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_nodes2, 5),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(hidden_nodes2, 1)
        self.to(device)
        
    
    def forward(self, x):
        features = self.feature_extractor(x)
        common = self.common(features)
        policy = self.actor(common)
        est_reward = self.critic(common)
        return policy, est_reward

class InverseModel(nn.Module):
  def __init__(self, n_actions, device='cpu'):
      super(InverseModel, self).__init__()
      self.fc = nn.Linear(64, n_actions)
      self.to(device)
      
  def forward(self, features):
      features = features.view(1, -1)
      action = self.fc(features) 
      return action

class ForwardModel(nn.Module):
    def __init__(self, n_actions, hidden_dims,device='cpu'):
        super(ForwardModel, self).__init__()
        self.fc = nn.Linear(hidden_dims+n_actions, hidden_dims)
        self.eye = torch.eye(n_actions)
        self.device = device
        self.to(device)
        
    def forward(self, action, features):
        action = action.to(self.device)
        features = features.to(self.device)
        x = torch.cat([self.eye[action].to(self.device), features], dim=-1) # (1, n_actions+hidden_dims)
        features = self.fc(x) # (1, hidden_dims)
        return features

class FeatureExtractor(nn.Module):
    def __init__(self, hidden_nodes1 = 128, hidden_nodes2 = 64,device='cpu'):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            # 2D conv layer
        
            create_conv_block(4, 16, 3),
            create_conv_block(16, 8, 3),
            create_conv_block(8, 4, 3),
        )


        self.flattering = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60516, hidden_nodes1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes1, hidden_nodes2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes2, 32),
        )
        self.to(device)
        
    def forward(self, x):
        y = torch.tanh(self.feature_extractor(x))
        y = self.flattering(y)
        return y

class PGLoss(nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()
    
    def forward(self, action_prob, reward):
        loss = -torch.mean(torch.log(action_prob+1e-6)*reward)
        return loss