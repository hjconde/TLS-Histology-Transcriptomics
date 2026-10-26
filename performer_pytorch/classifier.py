#!/usr/bin/env python
# encoding: utf-8
# Created by BIT09 at 2023/8/18
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_dim, dropout=0., dim=200, h_dim_1=256, h_dim_2=128, out_dim=10):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, dim))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=h_dim_1, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=h_dim_1, out_features=h_dim_2, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim_2, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


