
from data import get_aug_dataloader
import torch
import torch.nn as nn
from util import AverageMeter
import time

# Training
def train(cn, model, optimizer, t11, t21, t31, t41, t51, t12, t22, t32, t42, t52, t13, t23, t33, t43, t53, t14, t24, t34, t44, t54, criterion, wt):

    train_loss = AverageMeter()
    batch_time = AverageMeter()

    ctp = 0
    for param in model.named_parameters():
        if ctp <= cn:
            param[1].requires_grad = False
        ctp += 1

    model.train()

    end = time.time()

    for d11, d21, d31, d41, d51, d12, d22, d32, d42, d52, d13, d23, d33, d43, d53, d14, d24, d34, d44, d54 in zip(
            t11, t21, t31, t41, t51, t12, t22, t32, t42, t52, t13, t23, t33, t43, t53, t14, t24, t34, t44, t54):

        data = {}
        data[0] = [d11, d21, d31, d41, d51]
        data[1] = [d12, d22, d32, d42, d52]
        data[2] = [d13, d23, d33, d43, d53]
        data[3] = [d14, d24, d34, d44, d54]

        optimizer.zero_grad()

        loss = 0.0
        for dr in range(4):
            for dc in range(5):
                inputs, target = data[dr][dc][0].cuda(), data[dr][dc][1].cuda()
                outputs = model(inputs)
                loss_pre = criterion(outputs, target)
                loss += wt[dc] * loss_pre

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
    return


def dt(dir, num, bs):
    tder = get_aug_dataloader(dir + '/t' + str(num), is_validation=False,
                            batch_size=bs,
                            num_workers=8,
                            augs=4, shuffle=True)
    return tder


def selftrain(cn, model, dir_td, ep, wt, lr, bs):

    t11, t12, t13, t14 = dt(dir_td, 1, bs), dt(dir_td, 1, bs), dt(dir_td, 1, bs), dt(dir_td, 1, bs)
    t21, t22, t23, t24 = dt(dir_td, 2, bs), dt(dir_td, 2, bs), dt(dir_td, 2, bs), dt(dir_td, 2, bs)
    t31, t32, t33, t34 = dt(dir_td, 3, bs), dt(dir_td, 3, bs), dt(dir_td, 3, bs), dt(dir_td, 3, bs)
    t41, t42, t43, t44 = dt(dir_td, 4, bs), dt(dir_td, 4, bs), dt(dir_td, 4, bs), dt(dir_td, 4, bs)
    t51, t52, t53, t54 = dt(dir_td, 5, bs), dt(dir_td, 5, bs), dt(dir_td, 5, bs), dt(dir_td, 5, bs)

    print(len(t11.dataset), '  ', len(t21.dataset), '  ', len(t31.dataset), '  ', len(t41.dataset), '  ', len(t51.dataset),
          '  ', )

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    print('learning rate is ', lr)
    criterion = nn.CrossEntropyLoss()

    for ep_train in range(ep):
        train(cn, model, optimizer, t11, t21, t31, t41, t51, t12, t22, t32, t42, t52, t13, t23, t33, t43, t53, t14, t24, t34, t44, t54,
              criterion, wt)

    print(' ')

