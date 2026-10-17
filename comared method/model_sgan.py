import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
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
        #self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=0)
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
        logits = self.l_c2(h2)
        x_soft = F.softmax(logits, dim=1)
        return logits,x_soft,valid







class GAN(object):
    def __init__(self, label_size=100, batch_size=50, device='cpu', lrs=[0.0002, 0.0002], beta_1=0.5,
                 beta_2=0.999, momentum=0.9, pretrained=None):
        super(GAN, self).__init__()
        self.G = Generator().to(device)
        self.C1 = Classifier1().to(device)
        if pretrained:
            self.C1.load_state_dict(torch.load(pretrained).state_dict())
        self.C1.train()
        self.G.train()
        self.real_img = 1
        self.fake_img = 0
        self.optimG = optim.Adam(self.G.parameters(), lr=lrs[0], betas=(beta_1, beta_2))
        self.optimC1 = optim.SGD(self.C1.parameters(), lr=lrs[1], momentum=momentum)
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.device = device
        self.batch_size = batch_size
        self.label_size = label_size

    def train(self, data_loader1, data_loader2, epoch, log_every,save_dir_result):
        running_loss_G = 0.0
        running_loss_C1 = 0.0
        err1 = 0
        cor1 = 0
        for i, (((labeled_image1, real_label1, _, _), (unlabeled_image1, correctlabel1, _, _))) in enumerate(
               data_loader1, 0):
            noise = torch.rand(self.batch_size, 100).to(self.device)
            labeled_image1 = labeled_image1.to(self.device)
            unlabeled_image1 = unlabeled_image1.to(self.device)
            real_label1 = real_label1.to(self.device)
            gen_image = self.G(noise)
            r_label = torch.full((self.batch_size,), self.real_img, device=self.device)
            f_label = torch.full((self.batch_size,), self.fake_img, device=self.device)
            self.optimC1.zero_grad()
            est_labeled1, _,validla1 = self.C1(labeled_image1.data)
            _, _,validgen1 = self.C1(gen_image.data)
            _, _,validul1 = self.C1(unlabeled_image1.data)

            lossC1 = self.bce(validul1, r_label.reshape(-1, 1).float()) + self.bce(validgen1, f_label.reshape(-1,
                                                                                             1).float()) + self.bce(
                validla1, r_label.reshape(-1, 1).float())+self.ce(est_labeled1, real_label1)
            lossC1.backward()
            self.optimC1.step()
            self.optimG.zero_grad()
            gen_image = self.G(noise)
            _, _,validgen1 = self.C1(gen_image)
            lossG1 = self.bce(validgen1, r_label.reshape(-1, 1).float())
            lossG =  lossG1
            lossG.backward()
            self.optimG.step()
            running_loss_G += lossG.item()
            running_loss_C1 += lossC1.item()

        if (epoch + 1) % 10 - 1 == 0:
            result_file_path = os.path.join(save_dir_result, "errlabelnum.txt")
            with open(result_file_path, "a", encoding="utf-8") as f:
                f.write('%s :errall1 : %s    correct1 : %s        \n' % (
                epoch, err1, cor1))
        return err1

    def eval(self, dataloader):
        correct2 = 0
        total2 = 0
        self.C1.eval()
        with torch.no_grad():
            for data in dataloader:
                images, labels,index, img_path = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs, _ ,_= self.C1(images)
                _, predicted = torch.max(outputs.data, 1)
                total2 += labels.size(0)
                correct2 += (predicted == labels).sum().item()
        self.C1.train()
        return correct2/total2
    def save(self, epoch, fixed_noise, job_id):
        torch.save(self.C1,
                   "models/" + str(job_id) + "/C1_epoch_"  + "_label_" + str(self.label_size) + ".pt")
        torch.save(self.G,
                   "models/" + str(job_id) + "/G_epoch_"  + "_label_" + str(self.label_size) + ".pt")
    def print_model(self):
        print(self.C1)
        print(self.G)

    def imshow(self, fixed_noise,epoch,datanum):
        gen_img = self.G(fixed_noise)
        visualize(gen_img,epoch,datanum)
    def load(self, load_model):
        self.C1 = torch.load(load_model[0], map_location=self.device)
        self.G = torch.load(load_model[2], map_location=self.device)



