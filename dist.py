import os
import models
import torch
import torch.nn as nn
from data import get_aug_dataloader
import random

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
normlize = Normalize()


def feature(model, N, trainloader, f_dim):
    model.eval()
    onk = torch.zeros(N, f_dim).cuda()
    gt = torch.zeros(N, dtype=torch.int64).cuda()
    for batch_idx, (data, _, idx) in enumerate(trainloader):
        with torch.no_grad():
            data, idx = data.cuda(), idx.cuda()
            ft = model(data).detach()
            onk[idx, :] = normlize(ft)
    return onk.detach()

def dist(model, f_dim, dir_dt, ncl):

    model = model.cuda()
    model.eval()
    distm = {}

    for td in range(ncl):

        if os.path.exists(dir_dt + '/train_0' + str(td)) or os.path.exists(dir_dt + '/train_' + str(td)):
            if td < 10:
                trainloader = get_aug_dataloader(dir_dt + '/train_0' + str(td), is_validation=False,
                                                 batch_size=256,
                                                 num_workers=8,
                                                 augs=0, shuffle=True)
            else:
                trainloader = get_aug_dataloader(dir_dt + '/train_' + str(td), is_validation=False,
                                                 batch_size=256,
                                                 num_workers=8,
                                                 augs=0, shuffle=True)

            N = len(trainloader.dataset)
            onk = feature(model, N, trainloader, f_dim)

            dist = torch.zeros((N, N), dtype=torch.float64).cuda()
            onk = onk.softmax(dim=-1)
            for i in range(N):
                # onki = torch.zeros(1, f_dim).cuda()
                # onki[0] = onk[i]
                # dist[i] = torch.cosine_similarity(onki, onk, dim=1)
                dist[i] = torch.sqrt(torch.sum((onk[i] - onk) ** 2, 1))

            up = torch.triu(torch.ones(N, N), diagonal=1).cuda()
            low = torch.tril(torch.ones(N, N), diagonal=1).cuda()
            up = dist * up
            low = dist * low
            dist = up[:, 1:N] + low[:, 0:N - 1]

            distm[td] = dist

    return distm

