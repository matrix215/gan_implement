import pytorch_lightning as pl
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, random_split)
import torch.nn.functional as F
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

config = {
    'LATENT_SIZE':100,
    'HIDDEN_SIZE':256,
    'OUTPUT_SIZE':1,
    'EPOCHS':100,
    'LEARNING_RATE':0.0002,
    'BATCH_SIZE':128,
    'HEIGHT':28,
    'WIDTH':28,
    'CHANNEL':1,
    'SEED':42
}

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self,config,data_dir: str = '/content/'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        self.config = config

    def prepare_data(self):
        datasets.MNIST(self.data_dir,train=True,download=True)
        datasets.MNIST(self.data_dir,train=False,download=True)


    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)


    def train_dataloader(self):

class Generator(nn.Module):

  def __init__(self, config):
    super(Generator, self).__init__()

    # 입력층 노드 수
    self.inode = config["LATENT_SIZE"] # 28x28보다 작거나 같다? Z
    # 은닉층 노드 수
    self.hnode = config["HIDDEN_SIZE"]
    # 출력층 노드 수: 생성해야 하는 노드 수
    self.onode = config["HEIGHT"] * config['WIDTH'] # 28x28

    # 신경망 설계
    self.net = nn.Sequential(nn.Linear(self.inode, self.hnode, bias=True),
                             nn.LeakyReLU(),
                             nn.Dropout(0.1),
                             nn.Linear(self.hnode, self.hnode, bias=True),
                             nn.LeakyReLU(),
                             nn.Dropout(0.1),
                             nn.Linear(self.hnode, self.hnode, bias=True),
                             nn.LeakyReLU(),
                             nn.Dropout(0.1),
                             nn.Linear(self.hnode, self.onode, bias=True),
                             nn.Tanh())

  def forward(self, input_features):
    hypothesis = self.net(input_features)
    hypothesis = hypothesis.view(hypothesis.size(0),-1)
    return hypothesis

class Discriminator(nn.Module):

  def __init__(self, config):
    super(Discriminator, self).__init__()

    # 입력층 노드 수
    self.inode = config["HEIGHT"] * config['WIDTH'] # 28x28
    # 은닉층 노드 수
    self.hnode = config["HIDDEN_SIZE"]
    # 출력층 노드 수: 분류해야 하는 레이블 수
    self.onode = config["OUTPUT_SIZE"] # real? fake?

    # 신경망 설계
    self.net = nn.Sequential(nn.Linear(self.inode, self.hnode, bias=True),
                             nn.LeakyReLU(),
                             nn.Dropout(0.1),
                             nn.Linear(self.hnode, self.hnode, bias=True),
                             nn.LeakyReLU(),
                             nn.Dropout(0.1),
                             nn.Linear(self.hnode, self.hnode, bias=True),
                             nn.LeakyReLU(),
                             nn.Dropout(0.1),
                             nn.Linear(self.hnode, self.onode, bias=True),
                             nn.Sigmoid())
    
  def forward(self, input_features):
    hypothesis = self.net(input_features)
    return hypothesis
