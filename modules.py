import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class Flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mT


class ConvBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 filter_ch,
                 kernel_s=3,
                 dilation=1,
                 stride=1,
                 padding=0,
                 layers=4):
        super().__init__()

        self.conv_layer = nn.ModuleList()
        self.filter_size = filter_ch

        for i in range(layers):
            i = i % 3
            dilation = kernel_s ** i
            padding = (kernel_s * dilation - dilation) // 2
            block = nn.Sequential(
                Flip(),
                nn.Conv1d(in_channels=filter_ch,
                          out_channels=filter_ch,
                          kernel_size=kernel_s,
                          padding=padding,
                          dilation=dilation,
                          groups=filter_ch),
                Flip(),
                nn.LayerNorm(filter_ch),
                nn.ReLU(),
                Flip(),
                nn.Conv1d(filter_ch,
                          filter_ch,
                          1),
                Flip(),
                nn.LayerNorm(filter_ch),
                nn.ReLU())
            self.conv_layer.append(block)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        for i in range(len(self.conv_layer)):
            y = self.conv_layer[i](x)
            x = x + y
            x = self.drop(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 speaker_emb_size,
                 in_channels,
                 filter_channels,
                 n_convs,
                 hidden_dim,
                 voc_size):
        super().__init__()
        self.in_ch = in_channels
        self.filter_size = filter_channels
        self.voc_size = voc_size

        self.embedding = nn.Embedding(voc_size, in_channels, padding_idx=0)

        self.pre_sp = nn.Linear(speaker_emb_size, filter_channels)

        self.pre = nn.Linear(in_channels, filter_channels)
        self.conv_layer = ConvBlock(in_channels, filter_channels, layers=n_convs)
        self.proj = nn.Linear(filter_channels, filter_channels)
        self.fc = nn.Linear(filter_channels * 2, filter_channels * 2)

    def forward(self, x, s_emb):
        x = self.embedding(x)
        pre = F.relu(self.pre(x))
        x = self.conv_layer(pre)
        proj = self.proj(x)
        s_emb = self.pre_sp(s_emb)
        x_sp = torch.cat([proj, s_emb.unsqueeze(1).expand(-1, 379, -1)], dim=-1)
        x_sp = F.relu(self.fc(x_sp))
        return x_sp


class Decoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 filter_channels):
        super().__init__()
        self.pre = nn.Linear(hidden_dim * 4, filter_channels)
        self.fc = nn.Linear(filter_channels, filter_channels)
        self.proj = nn.Linear(filter_channels, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, z):
        z = F.relu(self.pre(z))
        z = self.drop(z)
        #z = F.relu(self.fc(z))
        z = self.proj(z)
        return z.squeeze(-1)
