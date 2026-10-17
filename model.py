
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from makedata import makedata
from mutualguide import MutualGuide
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def call_bn(bn, x):
    return bn(x)


class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.dropout_rate = 0.2
        self.c1=nn.Conv2d(3,32,kernel_size=11,stride=4, padding=0)
        self.c2=nn.Conv2d(32,64,kernel_size=5,stride=1, padding=0)
        self.c3=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=0)
        #self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=0)
        self.l_c1=nn.Linear(25600,1024)
        self.l_c2 = nn.Linear(1024, 3)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(64)
      #  self.bn4 = nn.BatchNorm2d(128)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h=x
        h=self.c1(h)
        h = F.leaky_relu((h), negative_slope=0.01)
        h = F.dropout2d(h, p=self.dropout_rate)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h = F.dropout2d(h, p=self.dropout_rate)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
       # h = self.c4(h)
        #h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h1 = h.view(h.size(0), -1)
        h2 = self.l_c1(h1)
        logits = self.l_c2(h2)
        return logits,h2

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.dropout_rate = 0.2
        self.c1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=0)
        self.c2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
                # self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=0)
        self.l_c1 = nn.Linear(25600, 1024)
        self.l_c2 = nn.Linear(1024, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        #self.bn4 = nn.BatchNorm2d(128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu((h), negative_slope=0.01)
        h = F.dropout2d(h, p=self.dropout_rate)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h = F.dropout2d(h, p=self.dropout_rate)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
                # h = self.c4(h)
                # h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h1 = h.view(h.size(0), -1)
        h2 = self.l_c1(h1)
        logits = self.l_c2(h2)
        return logits,h2




class Recyclable(object):
    def __init__(self, label_size=100, batch_size=50, device='cpu'):
        super(Recyclable, self).__init__()
        self.device = device
        self.C1 = Classifier1().to(device)
        self.C2 = Classifier2().to(device)
        self.C1.train()
        self.C2.train()
        self.optimC1 = optim.SGD(self.C1.parameters(), lr=0.0002, momentum=0.9)
        self.optimC2 = optim.SGD(self.C2.parameters(), lr=0.0002, momentum=0.9)
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.label_size = label_size

    def train(self, data_loader1,data_loader2, epoch,save_dir_result):
        running_loss_C1 = 0.0
        running_loss_C2 = 0.0
        err1 = 0
        err2 = 0
        cor1=0
        cor2=0
        num_batches = 1
        for i, (((labeled_image1, real_label1,_, _), (unlabeled_image1, correctlabel1,_, _)),((labeled_image2, real_label2,_, _), (unlabeled_image2,correctlabel2,_, _))) in enumerate(zip(data_loader1, data_loader2), 0) :
            labeled_image1 = labeled_image1.to(self.device)
            unlabeled_image1 = unlabeled_image1.to(self.device)
            real_label1 = real_label1.to(self.device)
            labeled_image2 = labeled_image2.to(self.device)
            unlabeled_image2 = unlabeled_image2.to(self.device)
            real_label2 = real_label2.to(self.device)
            # Create quasi-labels for the unlabeled image
            pseudo_ul_out1,_= self.C1(unlabeled_image1)  
            pseudo_ul_out2,_ = self.C2(unlabeled_image2)  
            pseudo_ul_label1 = torch.argmax(pseudo_ul_out1.data, 1).to(self.device)  
            pseudo_ul_label2 = torch.argmax(pseudo_ul_out2.data, 1).to(self.device)  


            for j in range(10):
                if pseudo_ul_label1[j]!=correctlabel1[j]:
                    err1=err1+1
                else:
                    cor1=cor1+1
            for k in range(10):
                if pseudo_ul_label2[k]!=correctlabel2[k]:
                    err2=err2+1
                else:
                    cor2 = cor2 + 1
            # Classifier #
            self.optimC1.zero_grad()
            est_labeled1,_= self.C1(labeled_image1.data)
            est_unlabeled1,_= self.C1(unlabeled_image1.data)
            lossC1 = self.ce(est_unlabeled1, pseudo_ul_label1) + self.ce(est_labeled1, real_label1)
            lossC1.backward()
            self.optimC1.step()
            self.optimC2.zero_grad()
            est_labeled2,_= self.C2(labeled_image2.data)
            est_unlabeled2,_= self.C2(unlabeled_image2.data)
            lossC2 = self.ce(est_unlabeled2, pseudo_ul_label2) + self.ce(est_labeled2, real_label2)
            lossC2.backward()
            self.optimC2.step()
            running_loss_C1 += lossC1.item()
            running_loss_C2 += lossC2.item()
            num_batches += 1
        result_file_pathloss = os.path.join(save_dir_result, "loss.txt")
        with open(result_file_pathloss, "a", encoding="utf-8") as f:
            f.write('%s  : loss :  %s  ,  %s \n' % (epoch, running_loss_C1/num_batches, running_loss_C2/num_batches))
        if (epoch + 1) % 10 == 0:
            result_file_path_sg = os.path.join(save_dir_result, "sg_errlabelnum.txt")
            with open(result_file_path_sg, "a", encoding="utf-8") as f:
                f.write('%s :sgerr1 : %s sgerr2 : %s     sgcor1 : %s sgcor2 :  %s        \n' % (epoch, err1, err2,cor1,cor2))
        return err1,err2

    def muguide(self,unlabel_loader1,unlabel_loader2,rootspseudo1,rootspseudo2,batch_size,epoch,save_dir_result,sam):
        defile1,er1,cor1 = MutualGuide(unlabel_loader2, self.C1, rootspseudo2,epoch,sam)
        defile2,er2,cor2 = MutualGuide(unlabel_loader1, self.C2, rootspseudo1,epoch,sam)
        for j in range(len(defile1)):
            if os.path.exists(defile1[j]):
                os.remove(defile1[j])
        for k in range(len(defile2)):
            if os.path.exists(defile2[k]):
                os.remove(defile2[k])
        print('building model...')
        rootdata_train1 = r"./data/labeldata1"
        rootdata_train2 = r"./data/labeldata2"
        rootextension1 = r"./data/unlabel1"
        rootextension2 = r"./data/unlabel2"
        roottest = r"./data/test"
        rootspseudo1 = r"./data/spseudolabel1"
        rootspseudo2 = r"./data/spseudolabel2"
        train_loader1, train_loader2, unlabel_loader1, unlabel_loader2, test, extension_dataset, train_dataset1, spseudo_loader1, spseudo_loader2 = makedata(
            rootdata_train1, rootdata_train2, rootextension1, rootextension2, roottest, rootspseudo1, rootspseudo2,
            batch_size)
        result_file_path_err = os.path.join(save_dir_result, "mg_errlabelnum.txt")
        with open(result_file_path_err, "a", encoding="utf-8") as f:
            f.write('%s :err1 : %s err2 : %s  correct1 : %s correct2 :  %s \n' % (epoch, er1, er2,cor1,cor2))

        return train_loader1, train_loader2, unlabel_loader1, unlabel_loader2, test, extension_dataset, train_dataset1, spseudo_loader1, spseudo_loader2,er1,er2
    def eval(self, dataloader):
        # correct1 = 0
        # total1 = 0
        # self.C1.eval()
        # with torch.no_grad():
        #     for data in dataloader:
        #         images, labels,index, img_path = data
        #         images = images.to(self.device)
        #         labels = labels.to(self.device)
        #         outputs= self.C1(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total1 += labels.size(0)
        #         correct1 += (predicted == labels).sum().item()
        # self.C1.train()
        # correct2 = 0
        # total2 = 0
        # self.C2.eval()
        # with torch.no_grad():
        #     for data in dataloader:
        #         images, labels,index, img_path = data
        #         images = images.to(self.device)
        #         labels = labels.to(self.device)
        #         outputs= self.C2(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total2 += labels.size(0)
        #         correct2 += (predicted == labels).sum().item()
        # self.C2.train()
        #
        correct = 0
        total = 0
        self.C1.eval()
        self.C2.eval()
        with torch.no_grad():
            for data in dataloader:
                images, labels,index, img_path = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs1,_= self.C1(images)
                outputs2,_ = self.C2(images)
                softmax1 = F.softmax(outputs1, dim=1)
                softmax2=F.softmax(outputs2, dim=1)
                softmax1, predicted1 = torch.max(softmax1.data, 1)
                softmax2, predicted2 = torch.max(softmax2.data, 1)
                predicted = torch.where(softmax1 > softmax2, predicted1, predicted2)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.C1.train()
        self.C2.train()
        return correct / total
    def evalsam(self, dataloader):
        same = 0
        total = 0
        self.C1.eval()
        self.C2.eval()
        with torch.no_grad():
            for data in dataloader:
                images, labels,index, img_path = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs1,_= self.C1(images)
                outputs2,_ = self.C2(images)
                softmax1 = F.softmax(outputs1, dim=1)
                softmax2=F.softmax(outputs2, dim=1)
                softmax1, predicted1 = torch.max(softmax1.data, 1)
                softmax2, predicted2 = torch.max(softmax2.data, 1)
                same+=(predicted1 == predicted2).sum().item()
                total += labels.size(0)
        self.C1.train()
        self.C2.train()
        return same / total
    def tnse(self, dataloader,dataloader2,save_dir_result,epoch):
        file_name = save_dir_result + 'tnse' + f"{epoch}" + '.jpeg'
        file_name2 = save_dir_result + 'c2tnse' + f"{epoch}" + '.jpeg'
        self.C1.eval()
        self.C2.eval()
        featureslist1 = []
        labels_list1 = []
        featureslist2 = []
        labels_list2 = []
        with torch.no_grad():
            for data in dataloader:
                images, labels,index, img_path = data
                images = images.to(self.device)
               # labels = labels.to(self.device)
                outputs1,features1= self.C1(images)
               # outputs2,features2 = self.C2(images)
                softmax1 = F.softmax(outputs1, dim=1)
               # softmax2=F.softmax(outputs2, dim=1)
                softmax1, predicted1 = torch.max(softmax1.data, 1)
               # softmax2, predicted2 = torch.max(softmax2.data, 1)
                featureslist1.append(features1.cpu().numpy())
                labels_list1.extend(predicted1.cpu().numpy())
        featureslist1 = np.concatenate(featureslist1, axis=0)
        labels = np.array(labels_list1)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,random_state=0)
        tsne_results = tsne.fit_transform(featureslist1)
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
        plt.colorbar()
        plt.title("t-SNE visualization of features")
        plt.savefig(file_name, dpi=300)
       # plt.show()
        plt.close('all') 
        with torch.no_grad():
            for data in dataloader2:
                images, labels,index, img_path = data
                images = images.to(self.device)
                labels = labels.to(self.device)
              #  outputs1,features1= self.C1(images)
                outputs2,features2 = self.C2(images)
               # softmax2 = F.softmax(outputs2, dim=1)
                softmax2=F.softmax(outputs2, dim=1)
               # softmax1, predicted1 = torch.max(softmax1.data, 1)
                softmax2, predicted2 = torch.max(softmax2.data, 1)
                featureslist2.append(features2.cpu().numpy())
                labels_list2.extend(predicted2.cpu().numpy())
        featureslist2 = np.concatenate(featureslist2, axis=0)
        labels2 = np.array(labels_list2)
        #tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
        tsne_results = tsne.fit_transform(featureslist2)
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels2, cmap='tab10')
        plt.colorbar()
        plt.title("t-SNE visualization of features")
        plt.savefig(file_name2, dpi=300)
        plt.close('all')  
        self.C1.train()
        self.C2.train()



    def save(self,  job_id):
        save_dir_result = f'./models/{job_id}'
        if not os.path.exists(save_dir_result):
            os.makedirs(save_dir_result)

        torch.save(self.C1,
                   "models/" + str(job_id) + "/C1_epoch_"  + "_label_" + str(self.label_size) + ".pt")
        torch.save(self.C2,
                   "models/" + str(job_id) + "/C2_epoch_"  + "_label_" + str(self.label_size) + ".pt")
    def print_model(self):
        print(self.C1)
        print(self.C2)
    def load(self, load_model):
        self.C1 = torch.load(load_model[0], map_location=self.device)
        self.C2 = torch.load(load_model[0], map_location=self.device)

