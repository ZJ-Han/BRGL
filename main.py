# -*- coding:utf-8 -*-
from model import *
from getback import getback,getbacklabel
from makedata import makedata
import torch
import math
from choose import choose
from itertools import cycle
from itertools import chain
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
classnum = 3
batch_size = 10 # labelnum>=batch size   10:32    5:
seed = 0           # manual seed
num_epochs =350  # number of epochs
lrC = 0.0002         # learning rate of classifier0.01
lrG = 0.0002       # learning rate of generator
lrD = 0.0002       # learning rate of discriminator
beta_1 = 0.5       # beta 1 for Adam optimizer
beta_2 = 0.999     # beta 2 for Adam optimizer
momentum = 0.9     # momentum for SGD optimizer
log_every = 1      # log frequency
device = "cuda:0"
exit_flag = False
if torch.cuda.is_available():
    device = torch.device(device)
else:
    device = torch.device("cpu")
torch.manual_seed(seed)
def main(datanum,job_id,samhold):
    global exit_flag
     # log id
    print('building model...')
    rootdata_train1 = r"./data/labeldata1"
    rootdata_train2 = r"./data/labeldata2"
    rootextension1 = r"./data/unlabel1"
    rootextension2 = r"./data/unlabel2"
    roottest = r"./data/test"
    rootspseudo1=r"./data/spseudolabel1"
    rootspseudo2 = r"./data/spseudolabel2"
    getback(rootspseudo1,rootextension1)
    getback(rootspseudo2,rootextension2)
    getbacklabel(rootdata_train2, rootextension2)
    getbacklabel(rootdata_train1, rootextension1)
    choose(classnum,datanum)
    loader1,loader2,unlabel_loader1,unlabel_loader2,test,extension_dataset,train_dataset1,spseudo_loader1,spseudo_loader2=makedata(rootdata_train1, rootdata_train2, rootextension1,rootextension2, roottest, rootspseudo1,rootspseudo2,batch_size)
    train_loader1 = chain(loader1, spseudo_loader1)
    train_loader2 = chain(loader2, spseudo_loader2)
    label_size=len(train_dataset1)
    Recyclable().print_model()
    Recycle = Recyclable(label_size=label_size,
                          batch_size=batch_size,
                          device=device
                          )
    zipped_loader1 = zip(cycle(train_loader1), unlabel_loader1)
    zipped_loader2 = zip(cycle(train_loader2), unlabel_loader2)
    best_correct=0
    er1=0
    er2=0


    proportion=10
    sam=0
    for epoch in range(num_epochs):
        save_dir_result = f'./result/{datanum}/{job_id}'
        if not os.path.exists(save_dir_result):
            os.makedirs(save_dir_result)
        err1,err2=Recycle.train(zipped_loader1,zipped_loader2, epoch,save_dir_result)
        if (epoch ) == proportion :
            errall1 = err1 + er1
            errall2 = err2 + er2
            result_file_path_err = os.path.join(save_dir_result, "allerrlabelnum.txt")
            with open(result_file_path_err, "a", encoding="utf-8") as f:
                f.write('%s :err1 : %s err2 : %s \n' % (epoch, errall1, errall2))
        train_loader1 = chain(loader1, spseudo_loader1)
        train_loader2 = chain(loader2, spseudo_loader2)
        zipped_loader1 = zip(cycle(train_loader1), unlabel_loader1)
        zipped_loader2 = zip(cycle(train_loader2), unlabel_loader2)
        if (epoch + 1) % 50 == 0:
            test1 = chain(train_loader1, unlabel_loader1)
            test2 = chain(train_loader2, unlabel_loader2)
            # sam = Recycle.evalsam(eval_model)
            a = Recycle.tnse(test1, test2, save_dir_result, epoch)
            # print('sampoint', sam)

        if (epoch + 1) % log_every == 0:
            # Evaluate the model on the validation set
            correct = Recycle.eval(test)
            print(correct)
            if correct > best_correct:
                best_correct = correct
                # Save the model
                Recycle.save(job_id)
                # Log the accuracy
                print('[', epoch + 1, ']', best_correct )


            result_file_path = os.path.join(save_dir_result, "result.txt")
            with open(result_file_path, "a", encoding="utf-8") as f:
                f.write('%s  :  %s   \n' % (epoch, correct))


        if (epoch+1) == proportion:
            eval_model = chain(train_loader1, unlabel_loader1)
            sam = Recycle.evalsam(eval_model)
            # if sam > samhold:
            #     exit_flag = True 
            #     break
            getback(rootspseudo1,rootextension1)
            getback(rootspseudo2,rootextension2)
            train_loader1, train_loader2, unlabel1, unlabel2, test, extension_dataset, train_dataset1, spseudo_loader1, spseudo_loader2 = makedata(
                rootdata_train1, rootdata_train2, rootextension1, rootextension2, roottest, rootspseudo1, rootspseudo2,
                batch_size)
            loader1, loader2, unlabel_loader1, unlabel_loader2, test, extension_dataset, train_dataset1, spseudo_loader1, spseudo_loader2,er1,er2=Recycle.muguide(unlabel1,unlabel2,rootspseudo1,rootspseudo2,batch_size,epoch,save_dir_result,sam)

            train_loader1 = chain(loader1, spseudo_loader1)
            train_loader2 = chain(loader2, spseudo_loader2)
            zipped_loader1 = zip(cycle(train_loader1), unlabel_loader1)
            zipped_loader2 = zip(cycle(train_loader2), unlabel_loader2)
            balance = 10 * math.log((epoch*sam), math.e)
            proportion=proportion+int(balance)
            #proportion=10 * math.log((epoch*sam), math.e)
    with open("bestacc.txt", "a", encoding="utf-8") as f:
        f.write('%s : acc : %s  \n' % (datanum, best_correct))
if __name__ == '__main__':
    datanum_list = [10]
    samhold=0.9995
    for i in range(len(datanum_list)):
        if exit_flag:
            break  
        main(datanum_list[i], i,samhold)
        exit_flag = False 
