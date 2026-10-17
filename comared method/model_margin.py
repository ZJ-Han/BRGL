import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from utils import *
from makedata import makedata
from mutualguide import MutualGuide
import random
import shutil
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 100
        self.fc = nn.Sequential(
            nn.Linear(100, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        x = self.deconv(x)
        return x

def call_bn(bn, x):
    return bn(x)


class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.dropout_rate = 0.2
        self.c1=nn.Conv2d(3,32,kernel_size=11,stride=4, padding=0)
        self.c2=nn.Conv2d(32,64,kernel_size=5,stride=1, padding=0)
        self.c3=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(25600,1024)
        self.l_c2 = nn.Linear(1024, 3)
        self.l_c3 = nn.Linear(1024, 1)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(64)
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
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h1 = h.view(h.size(0), -1)
        h2 = self.l_c1(h1)
        valid = self.l_c3(h2)
        valid = self.sigmoid(valid)
        return valid



class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.dropout_rate = 0.2
        self.c1=nn.Conv2d(3,32,kernel_size=11,stride=4, padding=0)
        self.c2=nn.Conv2d(32,64,kernel_size=5,stride=1, padding=0)
        self.c3=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(25600,1024)
        self.l_c2 = nn.Linear(1024, 3)
        self.l_c3 = nn.Linear(1024, 1)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(64)
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
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h1 = h.view(h.size(0), -1)
        h2 = self.l_c1(h1)
        logits = self.l_c2(h2)
        x_soft = F.softmax(logits, dim=1)
        return logits,x_soft


def inverted_cross_entropy_loss(pred, target, device, eps=1e-6):
    one_hot = torch.zeros(pred.shape)
    one_hot = one_hot.to(device)
    for i in range(pred.shape[0]):
        one_hot[i, target[i]] = 1
    loss = torch.mean(-torch.sum(torch.mul(one_hot, (torch.log(1 - pred + eps))), dim=1))
    return loss

class GAN(object):
    def __init__(self, label_size=100, batch_size=50, device='cpu', lrs=[0.0002, 0.0002], beta_1=0.5,
                 beta_2=0.999, momentum=0.9, pretrained=None):
        super(GAN, self).__init__()
        self.G = Generator().to(device)
        self.C1 = Classifier1().to(device)
        self.C2 = Classifier2().to(device)
        if pretrained:
            self.C1.load_state_dict(torch.load(pretrained).state_dict())
        if pretrained:
            self.C2.load_state_dict(torch.load(pretrained).state_dict())
        self.C1.train()
        self.C2.train()
        self.G.train()
        self.real_img = 1
        self.fake_img = 0
        self.optimG = optim.Adam(self.G.parameters(), lr=lrs[0], betas=(beta_1, beta_2))
        self.optimC1 = optim.SGD(self.C1.parameters(), lr=lrs[1], momentum=momentum)
        self.optimC2 = optim.SGD(self.C2.parameters(), lr=lrs[1], momentum=momentum)
        # losses
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.ice = inverted_cross_entropy_loss
        self.device = device
        self.batch_size = batch_size
        self.label_size = label_size

    def train(self, data_loader1, data_loader2, epoch, log_every):
        running_loss_G = 0.0
        running_loss_C1 = 0.0
        running_loss_C2 = 0.0
        err1 = 0
        err2 = 0
        for i, (((labeled_image1, real_label1, _, _), (unlabeled_image1, correctlabel1, _, _))) in enumerate(
               data_loader1, 0):
            noise = torch.rand(self.batch_size, 100).to(self.device)
            labeled_image1 = labeled_image1.to(self.device)
            unlabeled_image1 = unlabeled_image1.to(self.device)
            real_label1 = real_label1.to(self.device)
            gen_image = self.G(noise)
            pseudo_gen_out2, _ = self.C2(gen_image)
            pseudo_ul_out2, _ = self.C2(unlabeled_image1)
            pseudo_ul_label2 = torch.argmax(pseudo_ul_out2.data, 1).to(self.device)
            r_label = torch.full((self.batch_size,), self.real_img, device=self.device)
            f_label = torch.full((self.batch_size,), self.fake_img, device=self.device)
            self.optimC1.zero_grad()
            validla1 = self.C1(labeled_image1.data)
            validgen1 = self.C1(gen_image.data)
            validul1 = self.C1(unlabeled_image1.data)
            lossC1 = self.bce(validul1, r_label.reshape(-1, 1).float()) + self.bce(validgen1, f_label.reshape(-1,
                                                                                                              1).float()) + self.bce(
                validla1, r_label.reshape(-1, 1).float())
            lossC1.backward()
            self.optimC1.step()
            self.optimC2.zero_grad()
            est_labeled2, _ = self.C2(labeled_image1.data)
            _, est_gen_softmax2 = self.C2(gen_image.data)
            est_gen_label2 = torch.max(est_gen_softmax2, 1)[1]
            est_unlabeled2, _ = self.C2(unlabeled_image1.data)
            lossC2 = self.ce(est_unlabeled2, pseudo_ul_label2) + self.ce(est_labeled2, real_label1) + (
                        1 / (1 + np.exp(0.1 * (epoch - 100)))) * (self.ice(
                est_gen_softmax2, est_gen_label2, self.device))
            lossC2.backward()
            self.optimC2.step()
            self.optimG.zero_grad()
            gen_image = self.G(noise)
            validgen1 = self.C1(gen_image)
            lossG1 = self.bce(validgen1, r_label.reshape(-1, 1).float())
            pseudo_gen_out2, _ = self.C2(gen_image)
            pseudo_gen_label2 = torch.argmax(pseudo_gen_out2.data, 1).to(self.device)
            lossG2 = self.ce(pseudo_gen_out2, pseudo_gen_label2)
            lossG = ( lossG1 + lossG2)
            lossG.backward()
            self.optimG.step()
            running_loss_G += lossG.item()
            running_loss_C1 += lossC1.item()
            running_loss_C2 += lossC2.item()
        # if (epoch + 1) % 10 - 1 == 0:
        #     with open("sgerrlabelnum.txt", "a", encoding="utf-8") as f:
        #         f.write('%s :errall1 : %s errall2 : %s     correct1 : %s correct2 :  %s        \n' % (
        #         epoch, err1, err2, cor1, cor2))
        return err1, err2

    def eval(self, dataloader):
        correct1 = 0
        correct2 = 0
        total2 = 0
        self.C2.eval()
        with torch.no_grad():
            for data in dataloader:
                images, labels,index, img_path = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.C2(images)
                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()
        self.C2.train()
        return correct1,correct2/total2

    def save(self, epoch, fixed_noise, job_id):
        torch.save(self.C1,
                   "models/" + str(job_id) + "/C1_epoch_"  + "_label_" + str(self.label_size) + ".pt")
        torch.save(self.C2,
                   "models/" + str(job_id) + "/C2_epoch_"  + "_label_" + str(self.label_size) + ".pt")
        torch.save(self.G,
                   "models/" + str(job_id) + "/G_epoch_"  + "_label_" + str(self.label_size) + ".pt")
    def print_model(self):
        print(self.C1)
        print(self.C2)
        print(self.G)

    def imshow(self, fixed_noise,epoch,datanum):
        gen_img = self.G(fixed_noise)
        visualize(gen_img,epoch,datanum)

    def load(self, load_model):
        self.C1 = torch.load(load_model[0], map_location=self.device)
        self.G = torch.load(load_model[2], map_location=self.device)







