# -*- coding:utf-8 -*-
import math
from getback import getback,getbacklabel
from model import *
from choosecompare import choose
from itertools import chain
import torch.optim as optim
from mutualguidecompared import MutualGuide
from makedata import makedata
import torch
import torch.nn.functional as F
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 10  # batch size
seed = 0           # manual seed
classnum=3
num_epochs =350   # number of epochs
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
    save_dir_result = f'./selfresult/{datanum}'
    if not os.path.exists(save_dir_result):
        os.makedirs(save_dir_result)
    print('building model...')
    C1 = Classifier1()
    C1 = C1.to(device)
    optimizer1 = optim.SGD(C1.parameters(), lr=lrC, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    best_correct = 0
    er1=0
    cor1=0
    renum = 10
    num_batches1 = 1
    for epoch in range(num_epochs):
        running_loss1 = 0.0
        C1.train()
        # Training loop
        for i, (image, label,_,_) in enumerate(train_loader1, 0):
            image = image.to(device)
            label = label.to(device)
            optimizer1.zero_grad()
            outputs1,_= C1(image)
            loss1 = criterion(outputs1, label)
            loss1.backward()
            optimizer1.step()
            running_loss1 += loss1.item()
            num_batches1 += 1
        result_file_pathloss = os.path.join(save_dir_result, "loss.txt")
        with open(result_file_pathloss, "a", encoding="utf-8") as f:
            f.write('%s  : loss :  %s  \n' % (epoch, running_loss1/num_batches1))
        correct1 = 0
        total = 0
        C1.eval()
        # Evaluation loop
        with torch.no_grad():
            for data in test:
                images, labels,_,_ = data
                images = images.to(device)
                labels = labels.to(device)
                outputs1,_= C1(images)
                softmax1 = F.softmax(outputs1, dim=1)
                softmax1, predicted1 = torch.max(softmax1.data, 1)
                predicted = predicted1
                total += labels.size(0)
                correct1 += (predicted == labels).sum().item()
            result_file_path = os.path.join(save_dir_result, "result.txt")
            with open(result_file_path, "a", encoding="utf-8") as f:
                f.write('%s  : acc :  %s   \n' % (epoch, correct1/ total))
        print(correct1 / total)
        if (epoch) % 50 == 0:
            file_name = save_dir_result + 'tnse' + f"{epoch}" + '.jpeg'
            C1.eval()
            featureslist1 = []
            labels_list1 = []
            test1 = chain(loader1, spseudo_loader1)
            with torch.no_grad():
                for data in test1:
                    images, labels, index, img_path = data
                    images = images.to(device)
                    outputs1, features1 = C1(images)
                    softmax1 = F.softmax(outputs1, dim=1)
                    softmax1, predicted1 = torch.max(softmax1.data, 1)
                    featureslist1.append(features1.cpu().numpy())
                    labels_list1.extend(predicted1.cpu().numpy())
            featureslist1 = np.concatenate(featureslist1, axis=0)
            labels = np.array(labels_list1)
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
            tsne_results = tsne.fit_transform(featureslist1)
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
            plt.colorbar()
            plt.title("t-SNE visualization of features")
            plt.savefig(file_name, dpi=300)
            # plt.show()
            plt.close('all')  


        if (epoch+1) %50==0:
            renum = renum + 10
            if len(extension_dataset) > batch_size:
                unlabel1 =  unlabel_loader1
                defile1,er1,cor1 = MutualGuide(unlabel1, C1, rootspseudo1, epoch)
                for j in range(len(defile1)):
                    if os.path.exists(defile1[j]):
                        os.remove(defile1[j])
            print('Remaining unlabeled data : %s' % len(extension_dataset))
            result_file_path_err = os.path.join(save_dir_result, "errlabelnum.txt")
            with open(result_file_path_err, "a", encoding="utf-8") as f:
                f.write('%s :errmg1 : %s    correct1 : %s         \n' % (
                epoch, er1,  cor1))
            loader1, loader2, unlabel_loader1, unlabel_loader2, test, extension_dataset, train_dataset1, spseudo_loader1, spseudo_loader2 = makedata(
                rootdata_train1, rootdata_train2, rootextension1, rootextension2, roottest, rootspseudo1, rootspseudo2,
                batch_size)
        train_loader1 = chain(loader1, spseudo_loader1)

if __name__ == '__main__':
    for i in range(1):
        datanum=[10]
        main(datanum[i],i)