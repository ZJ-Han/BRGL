# -*- coding:utf-8 -*-
import random
import os
from dataloder import LoadData
from torch.utils.data import DataLoader
import re
def makedata(rootdata_train1,rootdata_train2,rootextension1,rootextension2,roottest,rootspseudo1,rootspseudo2,batch_size):
    train_list1, train_list2, extension_list1,extension_list2, test,spseudo_list1,spseudo_list2 = [], [], [], [],[],[],[]
    class_flag1 = -1
    for a, b, c in os.walk(rootdata_train1):
        b.sort()
        for i in range(int(len(c))):
            train_data1 = os.path.join(a, c[i]) + '\t' + str(class_flag1) + '\n'
            train_list1.append(train_data1)
        class_flag1 += 1
    class_flag3 = -1
    for a, b, c in os.walk(rootdata_train2):
        b.sort()
        for i in range(0, int(len(c))):
            train_data2 = os.path.join(a, c[i]) + '\t' + str(class_flag3) + '\n'
            train_list2.append(train_data2)
        class_flag3 += 1
    class_flag2 = -1
    for a, b, c in os.walk(rootextension1):
        b.sort()
        for i in range(0, int(len(c))):
            extension_data1 = os.path.join(a, c[i]) + '\t' + str(class_flag2) + '\n'
            extension_list1.append(extension_data1)
        class_flag2 += 1
    class_flag5 = -1
    for a, b, c in os.walk(rootextension2):
        b.sort()
        for i in range(0, int(len(c))):
            extension_data2 = os.path.join(a, c[i]) + '\t' + str(class_flag5) + '\n'
            extension_list2.append(extension_data2)
        class_flag5 += 1
    class_flag4 = -1
    for a, b, c in os.walk(roottest):
        b.sort()
        for i in range(0, int(len(c))):
            test_data4 = os.path.join(a, c[i]) + '\t' + str(class_flag4) + '\n'
            test.append(test_data4)
        class_flag4 += 1
    class_flags1 = -1
    for a, b, c in os.walk(rootspseudo1):
        b.sort()
        for i in range(0, int(len(c))):
            label1 = int(os.path.basename(a))
            spseudo_data1 = os.path.join(a, c[i]) + '\t' + str(label1) + '\n'
            spseudo_list1.append(spseudo_data1)
        class_flags1 += 1
    class_flags2 = -1
    for a, b, c in os.walk(rootspseudo2):
        b.sort()
        for i in range(0, int(len(c))):
            label2 = int(os.path.basename(a))
            spseudo_data2 = os.path.join(a, c[i]) + '\t' + str(label2) + '\n'
            spseudo_list2.append(spseudo_data2)
        class_flags2 += 1
    random.shuffle(train_list1)
    random.shuffle(train_list2)
    random.shuffle(extension_list1)
    random.shuffle(extension_list2)
    random.shuffle(test)
    random.shuffle(spseudo_list1)
    random.shuffle(spseudo_list2)
    with open('train1.txt', 'w', encoding='UTF-8') as f:
        for train_img1 in train_list1:
            f.write(str(train_img1))
    with open('train2.txt', 'w', encoding='UTF-8') as f:
        for train_img2 in train_list2:
            f.write(str(train_img2))
    with open('extension1.txt', 'w', encoding='UTF-8') as f:
        for test_img1 in extension_list1:
            f.write(test_img1)
    with open('extension2.txt', 'w', encoding='UTF-8') as f:
        for test_img3 in extension_list2:
            f.write(test_img3)
    with open('test.txt', 'w', encoding='UTF-8') as f:
        for test_img2 in test:
            f.write(test_img2)
    with open('spseudo1.txt', 'w', encoding='UTF-8') as f:
        for spseudo_img1 in spseudo_list1:
            f.write(spseudo_img1)
    with open('spseudo2.txt', 'w', encoding='UTF-8') as f:
        for spseudo_img2 in spseudo_list2:
            f.write(spseudo_img2)
    train_dataset1 = LoadData("train1.txt", True)
    train_dataset2 = LoadData("train2.txt", True)
    extension_dataset1 = LoadData("extension1.txt", False)
    extension_dataset2 = LoadData("extension2.txt", False)
    testload = LoadData("test.txt", False)
    spseudo_dataset1 = LoadData("spseudo1.txt", True)
    spseudo_dataset2= LoadData("spseudo2.txt", True)

    train_loader1 = DataLoader(dataset=train_dataset1, num_workers=0, pin_memory=False, batch_size=batch_size,
                               shuffle=False, drop_last=True)
    train_loader2 = DataLoader(dataset=train_dataset2, num_workers=0, pin_memory=False, batch_size=batch_size,
                               shuffle=False, drop_last=True)
    extension_loader1 = DataLoader(dataset=extension_dataset1, num_workers=0, pin_memory=False, batch_size=batch_size,
                                  shuffle=False, drop_last=True)
    extension_loader2 = DataLoader(dataset=extension_dataset2, num_workers=0, pin_memory=False, batch_size=batch_size,
                                  shuffle=False, drop_last=True)
    test = DataLoader(dataset=testload, num_workers=0, pin_memory=False, batch_size=batch_size,
                      shuffle=False)
    spseudo_loader1 = DataLoader(dataset=spseudo_dataset1, num_workers=0, pin_memory=False, batch_size=batch_size,
                      shuffle=False, drop_last=True)
    spseudo_loader2 = DataLoader(dataset=spseudo_dataset2, num_workers=0, pin_memory=False, batch_size=batch_size,
                      shuffle=False, drop_last=True)
    return train_loader1,train_loader2,extension_loader1 ,extension_loader2 ,test,extension_dataset1,train_dataset1,spseudo_loader1,spseudo_loader2