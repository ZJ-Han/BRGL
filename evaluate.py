# Evaluate the Model
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
def evaluate(test, model1, model2,model_str):
    print(('Evaluating %s...') % model_str)
    model1.eval()  
    correct1 = 0
    total1 = 0
    torch.set_printoptions(threshold=np.inf) 
    for i, (images, labels, indexes, img_path) in enumerate(test):
        images = Variable(images).cuda()
        logits1,feature = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()
    model2.eval()  
    correct2 = 0
    total2 = 0
    for i, (images, labels, indexes, img_path) in enumerate(test):
        images = Variable(images).cuda()
        logits2,feature = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)

    return acc1, acc2