import random
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import shutil
import math

def MutualGuide(extension_loader, model1,datas_root1,epoch,sam):
    defile1 = []
    img_path=np.array([1, ])
    softmax_list=np.array([1, ])
    label=np.array([1, ])
    pred=np.array([1, ])
    model1.eval()
    m = 0
    er1=0
    cor1=0
    torch.set_printoptions(threshold=np.inf)  # pycharm show all
    for i, (images, labels, indexes, img_paths) in enumerate(extension_loader):
        images = Variable(images).cuda()
        logits1,_ = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        softmax_list1, pred1 = torch.max(outputs1.data, 1)
        softmax_list1 = softmax_list1.data.cpu().numpy()
        pred1 = pred1.data.cpu().numpy()
        softmax_list = np.concatenate((softmax_list, softmax_list1))
        pred=np.concatenate((pred,pred1))
        img_path1 = np.array(img_paths)
        img_path = np.concatenate((img_path, img_path1 ))
        label1 = np.array(labels)
        label = np.concatenate((label, label1))

    img_path=img_path[1:,]
    softmax_list=softmax_list[1:,]
    label=label[1:,]
    pred=pred[1:,]

    intsort = np.argsort(-softmax_list)
    labeltrue = label[intsort ]
    softmax_list1sort = softmax_list[intsort ]
    labelsnew = pred[intsort ]
    img_path = img_path[intsort ]

    threshold=math.exp(-0.2 * sam)
    for iii in range(int(0.2*intsort.size) - 1):
       #if softmax_list1sort[iii] > (threshold-0.05) and softmax_list1sort[iii] < (0.9995):#mstar0.95-0.9995
      # if softmax_list1sort[iii] > (0.8) and softmax_list1sort[iii] < (1):  # mstar0.95-0.9995
            label = labelsnew[iii]
            fact1 = labeltrue[iii]
            chooseone = img_path[iii]
            file_name = datas_root1 + os.sep +  f"{label:02d}"  + os.sep + \
                             str(random.randint(0, 99999)) + 'va'+str(softmax_list1sort[iii])+ 'la' + f"{fact1:02d}" + '.jpeg'
            if os.path.exists(img_path[iii]):
                shutil.copyfile(chooseone, file_name)
                m=m+1
                defile1.append(img_path[iii])
                if fact1!=label:
                    er1=er1+1
                else:
                    cor1=cor1+1
    print('mgupdate : %d ' % m)
    print('mgerr label num: %d ' %er1)
    print('mgcor label num: %d ' % cor1)
    return defile1,er1,cor1