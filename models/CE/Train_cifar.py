from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--imb_factor', default=50, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, optimizer, trainloader):
    net.train()
    num_iter = (len(trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs, labels, _) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        # forward
        outputs = net(inputs)
        # loss = 交叉熵
        loss = F.cross_entropy(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印训练信息
        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-Loss: %.4f'
                % (args.dataset, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)                                   
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  




def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

# 日志文件
stats_log = open('./checkpoint/%s_%.1f_%s_stats.txt' % (args.dataset, args.r, args.noise_mode), 'w')
test_log = open('./checkpoint/%s_%.1f_%s_acc.txt' % (args.dataset, args.r, args.noise_mode), 'w')

# 数据加载
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, 
                          batch_size=args.batch_size, num_workers=5,
                          root_dir=args.data_path,
                          log=stats_log,
                          noise_file=f"{args.data_path}/CE_{args.dataset}{args.imb_factor}_{args.r:.2f}.json", imb_factor=args.imb_factor)

# 网络
print('| Building net')
net = create_model()
net = net.cuda()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

all_loss = []
for epoch in range(args.num_epochs + 1):
    # 调整学习率
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 数据加载器
    train_loader = loader.run('train')  
    test_loader = loader.run('test')                         
    print('Train Net')
    train(epoch,net,optimizer,train_loader)  
        
    # 每50轮保存一次 checkpoint
    if epoch % 50 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'/home/algroup/cyq/cp_pll/model/CE/checkpoint/CE_{args.dataset}{args.imb_factor}_r{args.r}_epoch{epoch}.pth')
        print("保存模型成功")

    test(epoch,net)