import os
import time
import torch

def self_label(onk, N, ncl):
    indx = 0
    nnp = N
    ave = torch.zeros(ncl).cuda()
    for i in range(ncl):
        ave[i] = i ** indx
    ave = ave * (nnp / torch.sum(ave))
    a = torch.zeros(ncl)
    a[0] = nnp
    max_std = torch.std(a)
    print('max_std:  ', max_std)
    del a

    lbs = torch.argmax(onk, 1)
    ftor = 1
    decay = 1
    decay_bound = 1e-15
    ct = torch.cat((torch.bincount(lbs), torch.zeros(ncl - torch.max(lbs + 1)).cuda()), 0)
    std = torch.std(ct)
    print('true std is: {:.5f}'.format(std))

    if std == 0:
        return lbs
    else:
        ct = ct - ave
        ct = ct / torch.max(ct)
        acct = 0
        # k = 0
        t_opt = time.time()
        while decay > decay_bound:

            rate = std / max_std
            decay = rate * (torch.max(onk) - torch.min(onk)) / ftor

            am = (ct * decay).cuda()
            onk -= am
            lbs = torch.argmax(onk, 1)

            ct_new = torch.cat((torch.bincount(lbs), torch.zeros(ncl - torch.max(lbs + 1)).cuda()), 0)
            std_new = torch.std(ct_new)

            if std_new < std:
                acct = 0
                std = std_new
                ct = ct_new - ave
                ct = ct / torch.max(ct)
            elif std_new > std:
                acct = 0
                ftor = ftor * 1.5
                onk += am
            else:
                acct += 1
                ct = ct_new - ave
                ct = ct / torch.max(ct)
            if acct == 10 or std == 0:
                break
    print('opt cost:  ', time.time() - t_opt)
    print('target std is: {:.5f}'.format(std))
    print('  ')
    return lbs


def lbs_update(N, model, trainloader, ncl, even):
    model.eval()
    onk = torch.zeros(N, ncl).cuda()
    print(onk.shape)
    t_nk = time.time()
    for batch_idx, (_, data, _, idx) in enumerate(trainloader):
        data = data.cuda()
        onk[idx, :] = model(data).detach()
    print('nk costs:   ', time.time() - t_nk)
    if even:
        selflabels = self_label(onk, N, ncl)
    else:
        selflabels = torch.argmax(onk, 1)
    return selflabels


import shutil


def mymovefile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)


from data_path import get_aug_dataloader

def cluster(model, dir, ncl, data_dir, even):
    trainloader = get_aug_dataloader(data_dir, is_validation=False,
                                     batch_size=256,
                                     num_workers=16,
                                     augs=0, shuffle=False)
    trainloader_div = get_aug_dataloader(data_dir, is_validation=False,
                                         batch_size=1,
                                         num_workers=16,
                                         augs=0, shuffle=False)

    N = len(trainloader.dataset)
    print('number of training pics is ', N)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    model = model.cuda()
    selflabels = lbs_update(N, model, trainloader, ncl, even)
    selflabels = selflabels.detach().cpu().numpy()
    for b_idx, (path, _, _, idx) in enumerate(trainloader_div):

        (filepath, tempfilename) = os.path.split(path[0])

        if selflabels[idx] < 10:
            mymovefile(path[0], dir + '/train_0' + str(selflabels[idx]) + '/train/' + filepath.split('/')[-1] + '/')
        else:
            mymovefile(path[0], dir + '/train_' + str(selflabels[idx]) + '/train/' + filepath.split('/')[-1] + '/')