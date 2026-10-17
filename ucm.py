# -*- coding:utf-8 -*-
import random
import os
from getback import getback
import torch
from modelucm import *
import argparse
import numpy as np
import datetime
from itertools import chain
import torch.optim as optim
import shutil
from dataloder import LoadData
from torch.utils.data import DataLoader
from train import train
from evaluate import evaluate
from muucm import MutualGuide
import matplotlib.pyplot as plt
from makedata import makedata
import torch
from itertools import cycle
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from accuracy import accuracy
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 50    # batch size
seed = 0           # manual seed
num_epochs =600   # number of epochs
lrC = 0.0002         # learning rate of classifier
lrG = 0.0002       # learning rate of generator
lrD = 0.0002       # learning rate of discriminator
beta_1 = 0.5       # beta 1 for Adam optimizer
beta_2 = 0.999     # beta 2 for Adam optimizer
momentum = 0.8     # momentum for SGD optimizer
log_every = 1      # log frequency
job_id = 1         # log id
device = "cuda:0"  # device
if torch.cuda.is_available():
    device = torch.device(device)
else:
    device = torch.device("cpu")
#warnings.filterwarnings("ignore")
torch.manual_seed(seed)
def main():
    print('building model...')
    rootdata_train1 = r"./data/labeldata1"
    rootdata_train2 = r"./data/labeldata2"
    rootextension1 = r"./data/unlabel1"
    rootextension2 = r"./data/unlabel2"
    roottest = r"./data/test"
    rootspseudo1=r"./data/spseudolabel1"
    rootspseudo2 = r"./data/spseudolabel2"

    loader1,loader2,unlabel_loader1,unlabel_loader2,test,extension_dataset,train_dataset1,spseudo_loader1,spseudo_loader2=makedata(rootdata_train1, rootdata_train2, rootextension1,rootextension2, roottest, rootspseudo1,rootspseudo2,batch_size)

    train_loader1 = chain(loader1, spseudo_loader1)
    train_loader2 = chain(loader2, spseudo_loader2)
    print('building model...')

    C1 =CNN1(input_channel=3, n_outputs=21)
    C1 = C1.to(device)
    C2 = CNN2(input_channel=3, n_outputs=21)
    C2 = C2.to(device)
    optimizer1 = optim.SGD(C1.parameters(), lr=lrC, momentum=momentum)
    optimizer2 = optim.SGD(C2.parameters(), lr=lrC, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        C1.train()
        # Training loop
        for i, (image, label,_,_) in enumerate(train_loader1, 0):
            image = image.to(device)
            label = label.to(device)
            optimizer1.zero_grad()
            outputs1= C1(image)
            loss1 = criterion(outputs1, label)
            loss1.backward()
            optimizer1.step()
            running_loss += loss1.item()
            if (i + 1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        C2.train()
        for i, (image, label,_,_) in enumerate(train_loader2, 0):
            image = image.to(device)
            label = label.to(device)
            optimizer2.zero_grad()
            outputs2= C2(image)
            loss2 = criterion(outputs2, label)
            loss2.backward()
            optimizer2.step()
            running_loss += loss2.item()
            if (i + 1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


        correct1 = 0
        correct2 = 0
        total = 0
        C1.eval()
        C2.eval()
        # Evaluation loop
        with torch.no_grad():
            for data in test:
                images, labels,_,_ = data
                images = images.to(device)
                labels = labels.to(device)
                out1= C1(images)
                out2= C2(images)
                outputs1 = F.softmax(out1, dim=1)
                outputs2 = F.softmax(out2, dim=1)
                _, predicted1 = torch.max(outputs1, 1)
                _, predicted2 = torch.max(outputs2, 1)
                total += labels.size(0)
                correct1 += (predicted1 == labels).sum().item()
                correct2 += (predicted2 == labels).sum().item()
        with open("result.txt", "a", encoding="utf-8") as f:
            f.write('%s  : acc :  %s  , %s  \n' % (epoch, correct1/ total, correct2/ total))
        print(correct1 / total,correct2 / total)








        if epoch%50==0:
            getback(rootspseudo1, rootextension2)
            getback(rootspseudo2, rootextension1)
            train_loader1, train_loader2, unlabel1, unlabel2, test, extension_dataset, train_dataset1, spseudo_loader1, spseudo_loader2 = makedata(
                rootdata_train1, rootdata_train2, rootextension1, rootextension2, roottest, rootspseudo1, rootspseudo2,
                batch_size)



            if len(extension_dataset) > batch_size:
                # unlabel2 = chain(spseudo_loader1, unlabel_loader2)
                # unlabel1 = chain(spseudo_loader2, unlabel_loader1)
                # unlabel2 =  unlabel_loader2
                # unlabel1 =  unlabel_loader1
                defile1 = MutualGuide(unlabel2, C1, rootspseudo1, epoch)
                defile2 = MutualGuide(unlabel1 , C2, rootspseudo2, epoch)
                for j in range(len(defile1)):
                    if os.path.exists(defile1[j]):
                        os.remove(defile1[j])
                for k in range(len(defile2)):
                    if os.path.exists(defile2[k]):
                        os.remove(defile2[k])
            print('Remaining unlabeled data : %s' % len(extension_dataset))
            loader1, loader2, unlabel_loader1, unlabel_loader2, test, extension_dataset, train_dataset1, spseudo_loader1, spseudo_loader2 = makedata(
                rootdata_train1, rootdata_train2, rootextension1, rootextension2, roottest, rootspseudo1, rootspseudo2,
                batch_size)
            train_loader1 = chain(loader1, spseudo_loader1)
            train_loader2 = chain(loader2, spseudo_loader2)
            # C1 = Classifier1()
            # C1 = C1.to(device)
            # C2 = Classifier2()
            # C2 = C2.to(device)
            # optimizer1 = optim.SGD(C1.parameters(), lr=lrC, momentum=momentum)
            # optimizer2 = optim.SGD(C2.parameters(), lr=lrC, momentum=momentum)
            # criterion = nn.CrossEntropyLoss()
        train_loader1 = chain(loader1, spseudo_loader1)
        train_loader2 = chain(loader2, spseudo_loader2)



if __name__ == '__main__':
    main()
