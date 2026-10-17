import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from accuracy import accuracy
# Train the Model
def train(train_loader, epoch, model, optimizer,model_str,num_iter_per_epoch):
    #print('Training %s...' % model_str)
    train_total = 0
    train_correct = 0
    loss_fn = torch.nn.BCELoss()  
    for i, (images, labels, indexes, img_path) in enumerate(train_loader):
        if i > num_iter_per_epoch:
            break
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        logits,feature= model(images)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec
        loss1 = F.cross_entropy(logits, labels, reduce=False)  
        loss = torch.sum(loss1) / (len(loss1))
        d_real_loss = loss
        optimizer.zero_grad()  
        d_real_loss.backward()  
        optimizer.step()  
        # if ((i + 1) % 16) == 0:
        #     print(loss.item())
        #     print('Epoch [%d/%d],Training Accuracy: %.4f %% Loss: %.4f %%' % (epoch + 1, 128, prec, loss.item()))
    train_acc = float(train_correct) / float(train_total)
    return train_acc, loss.item()