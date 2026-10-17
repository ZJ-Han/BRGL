import torch.nn as nn
import torch.nn.functional as F
import torch
def call_bn(bn, x):
    return bn(x)
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CNN1(nn.Module):
    def __init__(self, input_channel=3, n_outputs=21, dropout_rate=0.2, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN1, self).__init__()
        self.c1=nn.Conv2d(input_channel,16,kernel_size=5,stride=2, padding=0)
        self.c2=nn.Conv2d(16,32,kernel_size=5,stride=2, padding=0)
        self.c3=nn.Conv2d(32,64,kernel_size=5,stride=2, padding=0)
        self.c4 = nn.Conv2d(64, 128, kernel_size=5, stride=2,padding=0)  
        self.c5 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0)
        self.c6 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=0)
        self.fc = nn.Linear(28544, 1)
        self.l_c1=nn.Linear(512,n_outputs)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,):
        h=x
        h=self.c1(h)
        h = F.leaky_relu((h), negative_slope=0.01)  
        h = F.dropout2d(h, p=self.dropout_rate)  
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.dropout2d(h, p=self.dropout_rate)  
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h3 = h.view(h.size(0), -1)
        logits = self.l_c1(h3)

        return logits



class CNN2(nn.Module):
    def __init__(self, input_channel=3, n_outputs=21, dropout_rate=0.2, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN2, self).__init__()
        self.c1=nn.Conv2d(input_channel,16,kernel_size=5,stride=2, padding=0)
        self.c2=nn.Conv2d(16,32,kernel_size=5,stride=2, padding=0)
        self.c3=nn.Conv2d(32,64,kernel_size=5,stride=2, padding=0)
        self.c4 = nn.Conv2d(64, 128, kernel_size=5, stride=2,padding=0)  
        self.c5 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0)
        self.c6 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=0)
        self.fc = nn.Linear(28544, 1)
        self.l_c1=nn.Linear(512,n_outputs)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,):
        h=x
        h=self.c1(h)
        h = F.leaky_relu((h), negative_slope=0.01)  
        h = F.dropout2d(h, p=self.dropout_rate)  
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.dropout2d(h, p=self.dropout_rate)  
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h3 = h.view(h.size(0), -1)
        logits = self.l_c1(h3)
        return logits


