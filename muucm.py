import random
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import shutil
import math

def MutualGuide(extension_loader, model1,datas_root1,epoch):
    defile1 = []
    model1.eval()  
    m = 0
    er1=0
    torch.set_printoptions(threshold=np.inf)  
    for i, (images, labels, indexes, img_path) in enumerate(extension_loader):
        images = Variable(images).cuda()
       # images = images.unsqueeze(0)
        out = model1(images)
        outputs1 = F.softmax(out , dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        softmax_list1 = _
        #################################################################
        softmax_list1 = softmax_list1.data.cpu().numpy()  
        int1 = np.argsort(-softmax_list1)
        labeltrue1 = labels[int1]
        softmax_list1sort = softmax_list1[int1]
        labelsnew1 = pred1[int1]
        labelsnew1 = labelsnew1.cuda().data.cpu().numpy()
        labeltrue1 = labeltrue1.cuda().data.cpu().numpy()
        img_path1 = np.array(img_path)
        img_path1 = img_path1[int1]
        for iii in range(int( len(int1)) - 1):
            confidence_interval=0.2+epoch*0.002
            if softmax_list1sort[iii] > (0.85) and softmax_list1sort[iii] < (1):#mnist0.93-0.98
                label = labelsnew1[iii]
                fact1 = labeltrue1[iii]
                chooseone = img_path1[iii]
                file_name = datas_root1 + os.sep +  f"{label:02d}"  + os.sep + \
                            str(random.randint(0, 99999)) + 'va'+str(softmax_list1sort[iii])+ 'la' + f"{fact1:02d}" + '.jpg'
                if os.path.exists(img_path1[iii]):
                    shutil.copyfile(chooseone, file_name)
                    #print('update1 : %d : %s' % (iii, img_path1[iii]))
                    m=m+1
                    defile1.append(img_path1[iii])
                if fact1!=label:
                    er1=er1+1
                #     print('update : %d : %s : %s' % (iii, img_path1[iii], str(softmax_list1sort[iii])))





    print('update : %d ' % m)
    print('err label num: %d ' %er1)
    return defile1