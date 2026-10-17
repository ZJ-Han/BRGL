# -*- coding:utf-8 -*-
from model_margin import *
from getback import getback,getbacklabel
from makedata import makedata
import torch
from choosecompare import choose
from itertools import cycle
from itertools import chain
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
classnum=3
batch_size = 10  # labelnum>=batch size   10:32    5:
seed = 0           # manual seed
num_epochs = 50   # number of epochs
lrC = 0.0002         # learning rate of classifier0.01
lrG = 0.0002       # learning rate of generator
lrD = 0.0002       # learning rate of discriminator
beta_1 = 0.5       # beta 1 for Adam optimizer
beta_2 = 0.999     # beta 2 for Adam optimizer
momentum = 0.9     # momentum for SGD optimizer
log_every = 1      # log frequency
job_id = 1         # log id
device = "cuda:0"  # device
if torch.cuda.is_available():
    device = torch.device(device)
else:
    device = torch.device("cpu")
torch.manual_seed(seed)
def main(datanum,job_id):
    save_dir_result = f'./marginganresult/{datanum}'
    if not os.path.exists(save_dir_result):
        os.makedirs(save_dir_result)
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
    train_loader2 = chain(loader2, spseudo_loader2)
    label_size=len(train_dataset1)
    GAN().print_model()
    marginGAN = GAN(label_size=label_size,
                          batch_size=batch_size,
                          device=device,
                          lrs=[lrG, lrC],
                          beta_1=beta_1,
                          beta_2=beta_2,
                          momentum=momentum
                          )
    zipped_loader1 = zip(cycle(train_loader1), unlabel_loader1)
    zipped_loader2 = zip(cycle(train_loader2), unlabel_loader2)
    best_correct=0
    for epoch in range(num_epochs):
        marginGAN.train(zipped_loader1,zipped_loader2, epoch, log_every)
        train_loader1 = chain(loader1, spseudo_loader1)
        train_loader2 = chain(loader2, spseudo_loader2)
        zipped_loader1 = zip(cycle(train_loader1), unlabel_loader1)
        zipped_loader2 = zip(cycle(train_loader2), unlabel_loader2)
        if (epoch + 1) % log_every == 0:
            correct1,correct2 = marginGAN.eval(test)
            print(correct1,correct2)
            result_file_path = os.path.join(save_dir_result, "result.txt")
            with open(result_file_path, "a", encoding="utf-8") as f:
                f.write('%s : acc : %s         \n' % (
                    epoch, correct1))
if __name__ == '__main__':
    for i in range(6):
        datanum=[ 1,2,5,10,15,20]
        main(datanum[i], i)