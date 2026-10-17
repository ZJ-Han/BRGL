# -*- coding:utf-8 -*-
from getback import getback,getbacklabel
from model import *
from choosecompare import choose
from itertools import chain
import torch.optim as optim
from mutualguide import MutualGuide
from makedata import makedata
import torch
import torch.nn.functional as F
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 10  # batch size
seed = 0           # manual seed
classnum=3
num_epochs =50  # number of epochs
lrC = 0.0002         # learning rate of classifier
lrG = 0.0002       # learning rate of generator
lrD = 0.0002       # learning rate of discriminator
beta_1 = 0.5       # beta 1 for Adam optimizer
beta_2 = 0.999     # beta 2 for Adam optimizer
momentum = 0.9     # momentum for SGD optimizer
log_every = 1      # log frequency
device = "cuda:0"  # device
if torch.cuda.is_available():
    device = torch.device(device)
else:
    device = torch.device("cpu")
torch.manual_seed(seed)
def main(datanum,job_id):
    print('building model...')
    rootdata_train1 = r"../data/labeldata1"
    rootdata_train2 = r"../data/labeldata2"
    rootextension1 = r"../data/unlabel1"
    rootextension2 = r"../data/unlabel2"
    roottest = r"../data/test"
    rootspseudo1=r"../data/spseudolabel1"
    rootspseudo2 = r"../data/spseudolabel2"
    getback(rootspseudo1,rootextension1)
    getback(rootspseudo2,rootextension2)
    getbacklabel(rootdata_train2, rootextension2)
    getbacklabel(rootdata_train1, rootextension1)
    choose(classnum,datanum)
    loader1,loader2,unlabel_loader1,unlabel_loader2,test,extension_dataset,train_dataset1,spseudo_loader1,spseudo_loader2=makedata(rootdata_train1, rootdata_train2, rootextension1,rootextension2, roottest, rootspseudo1,rootspseudo2,batch_size)
    train_loader1 = chain(loader1, spseudo_loader1)
    save_dir_result = f'./cnnresult/{datanum}'
    if not os.path.exists(save_dir_result):
        os.makedirs(save_dir_result)
    print('building model...')
    C1 = Classifier1()
    C1 = C1.to(device)
    optimizer1 = optim.SGD(C1.parameters(), lr=lrC, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    best_correct = 0
    num_batches1 = 1
    for epoch in range(num_epochs):
        running_loss1 = 0.0
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
            running_loss1 += loss1.item()
            num_batches1 += 1

        result_file_pathloss = os.path.join(save_dir_result, "loss.txt")
        with open(result_file_pathloss, "a", encoding="utf-8") as f:
            f.write('%s  : loss :  %s   \n' % (epoch, running_loss1/num_batches1))
        correct1 = 0
        total = 0
        C1.eval()
        # Evaluation loop
        with torch.no_grad():
            for data in test:
                images, labels,_,_ = data
                images = images.to(device)
                labels = labels.to(device)
                outputs1= C1(images)
                softmax1 = F.softmax(outputs1, dim=1)
                softmax1, predicted = torch.max(softmax1.data, 1)
                total += labels.size(0)
                correct1 += (predicted == labels).sum().item()
            result_file_path = os.path.join(save_dir_result, "result.txt")
            with open(result_file_path, "a", encoding="utf-8") as f:
                f.write('%s  : acc :  %s   \n' % (epoch, correct1/ total))
        print(correct1 / total)
        train_loader1 = chain(loader1, spseudo_loader1)
    result_file = os.path.join(save_dir_result, "bestacc.txt")
    with open(result_file, "a", encoding="utf-8") as f:
        f.write('%s : acc : %s  \n' % (datanum, best_correct))

if __name__ == '__main__':
    for i in range(6):
        datanum=[ 1,2,5,10,15,20]
        main(datanum[i],i)