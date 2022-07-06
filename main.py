import torch
import os
import argparse
import torchvision
from util import setup_runtime
import models
import models_eval
import torchvision.transforms as tfs
import numpy as np

from cluster import cluster
from dist import dist
from class_topk import classifier
from train import selftrain

from eval_utils import pre_tar, humgarian_acc


parser = argparse.ArgumentParser(description='PyTorch Implementation')
parser.add_argument('--arch', default='resnetv1_18', type=str, help='architecture')
parser.add_argument('--ncl', default=21, type=int, help='number of clusters')
parser.add_argument('--hc', default=1, type=int, help='number of heads')
parser.add_argument('--nk', default=8, type=int, help='number of workers')

parser.add_argument('--device', default="0", type=str, help='cuda device')
parser.add_argument('--id', default='uc', type=str, help='dataset')
parser.add_argument('--cl', default=21, type=int, help='number of class')
parser.add_argument('--rep', default=1, type=int, help='number of class')
parser.add_argument('--dy', default=0.03, type=float, help='learning rate')

parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--epochs', default=120, type=int, help='number of epochs to train')
parser.add_argument('--bs', default=80, type=int, metavar='BS', help='batch size')

parser.add_argument('--datadir', default='/raid/lql/data/rs/', type=str)
parser.add_argument('--ck', default='uc_cl_best.pth', type=str, help='model.pth')
args = parser.parse_args()

print(parser)

setup_runtime(2, [args.device])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################################################################################################  MODEL

print('==> Building model..')
numc = [args.cl] * args.hc
model = models.__dict__[args.arch](num_classes=numc)
model_ft = models_eval.__dict__[args.arch]([args.ncl])

pth = '/raid/lql/models_saved/' + args.ck

# dir_model = '/raid/lql/clustering/model/' + args.id + '/'
# pth = dir_model + 'clster.pth'

model.load_state_dict(torch.load(pth))
model.to(device)

model_ft.load_state_dict(torch.load(pth))
model_ft.to(device)

#######################################################################################################  DATA

class DataSet(torch.utils.data.Dataset):

    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)

transform_test = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = DataSet(torchvision.datasets.ImageFolder(args.datadir + args.id + '/train', transform_test))
trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.nk,
        pin_memory=True,
        drop_last=False)

N = len(trainloader.dataset)
print('Number of images is ', N)

#######################################################################################################  EVALUATION

def eval(model):
    predictions, targets = pre_tar(model, trainloader, args.cl)
    acc = humgarian_acc(predictions, targets, args.cl)
    return acc

eval(model)

#######################################################################################################  TRAIN

dir_data = '/raid/lql/clustering/data/' + args.id + '/'
dir_model = '/raid/lql/clustering/model/' + args.id + '/'

pth_md = dir_model + 'clster.pth'
dir_cluster = dir_data + 'cluster/'
dir_dist = dir_data + 'dist/'
dir_class = dir_data + 'class/'

def wt_arrage(decay):
    wt = torch.ones(5)
    for w in range(5):
        wt[w] = 1 - (w ** decay) / (5 ** decay)
    return wt

def adjust_learning_rate(epoch):
    if epoch <= int(args.epochs*0.75):
        lr = args.lr
    else:
        lr = args.lr * 0.1
    return lr


acc_best = 0.0
dcy = torch.zeros(args.epochs)
acc_m = []

percent = torch.tensor([150, 350, 550, 750, 950])

for round in range(2):

    if round > 0:
        percent = torch.tensor([150, 350, 550, 750, 950]) + 25 * torch.tensor([1, 1, 1, 1, 1])

    decay = 0.0

    for epoch in range(args.epochs):

        lr = adjust_learning_rate(epoch)

        decay = decay + args.dy
        dcy[epoch] = decay

        print(
            '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
            'round: ', round, 'epoch: ', epoch)

        print('model saved to ', pth_md)

        if epoch < (90 * args.epochs) // 100:
            cluster(model, dir_cluster, args.cl, args.datadir+args.id, even=True)
        else:
            cluster(model, dir_cluster, args.cl, args.datadir+args.id, even=False)

        if round < 2:
            rep = True
        else:
            rep = False

        if rep:
            if epoch == 0 and round == 0:
                model_ft.load_state_dict(torch.load(pth))
            else:
                model_ft.load_state_dict(torch.load(pth_md))
            model_ft.to(device)
            distm = dist(model_ft, 512, dir_cluster, args.cl)
        else:
            distm = dist(model, cl, dir_cluster, args.cl)

        step = 2

        classifier(step, dir_cluster, dir_class, percent, args.cl, distm)

        ep = 2
        wt = wt_arrage(decay)
        selftrain(-1, model, dir_class, ep, wt, lr, args.bs)
        print('For rate ', args.dy, ' : ', percent, ',   ',  wt)

        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        torch.save(model.state_dict(), pth_md)

        acc = eval(model)
        if acc > acc_best:
            acc_best = acc
            torch.save(model.state_dict(), dir_model + 'clster_best.pth')

        print('--------------------------------------------------------------------------------------------------------------------------------------acc', acc * 100, '     best acc', acc_best*100)

        acc_m.append(acc)

        torch.save(acc_m, dir_model + '/acc_m.pt')
        torch.save(dcy, dir_model + '/dcy.pt')
        print(' ')