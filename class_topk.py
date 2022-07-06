import os
import numpy as np
import torch
from data_path import get_aug_dataloader
import shutil


def st(dist):
    dist_mid, _ = torch.median(dist, 1)
    dist_mean = torch.mean(dist, 1)
    _, idx = torch.sort(dist_mean + dist_mid, descending=False)
    return dist[idx], idx


def mm(dst):
    mid, _ = torch.median(dst, 1)
    mn = torch.mean(dst, 1)
    qm = mid + mn
    return qm


def sd_idx(pct, dist, unt):
    N = len(dist)
    dist, _ = torch.sort(dist, descending=False, dim=1)
    dist, nidx = st(dist)

    mk = pct * (N-1) // 1000
    num = (N - 1) // unt
    start_num = int(num*0.3)
    len_num = int(num*0.4)

    #print('train_' + str(td), '  mk', mk, '  num', N, '  unit', unt)

    midx = torch.empty(0, dtype=torch.int32).cuda()
    midx_ = [i for i in range(N)]
    didx = nidx - torch.tensor(midx_).cuda()
    for s in range(len_num):
        qm = mm(dist[:, 0: unt * start_num + unt * s])
        _, idx = torch.topk(qm, mk, largest=False)
        midx = torch.cat((midx, idx), 0)

    ct = torch.bincount(midx)
    _, midx = torch.topk(ct, mk, largest=True)

    fidx = torch.zeros(len(midx), dtype=torch.int32)
    for m in range(len(midx)):
        fidx[m] = midx[m] + didx[midx[m]]

    return fidx


def mymovefile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)


def classifier(unit, dir_dt, dir, percent, ncl, distm):
    if os.path.exists(dir):
        shutil.rmtree(dir)

    for td in range(ncl):
        if os.path.exists(dir_dt + '/train_0' + str(td)) or os.path.exists(dir_dt + '/train_' + str(td)):
            if td < 10:
                trainloader_div = get_aug_dataloader(dir_dt + '/train_0' + str(td),
                                                     is_validation=False,
                                                     batch_size=1,
                                                     num_workers=16,
                                                     augs=0, shuffle=False)
            else:
                trainloader_div = get_aug_dataloader(dir_dt + '/train_' + str(td),
                                                     is_validation=False,
                                                     batch_size=1,
                                                     num_workers=16,
                                                     augs=0, shuffle=False)

            pidx_1 = sd_idx(percent[0], distm[td], unit).cpu().numpy()
            pidx_2 = sd_idx(percent[1], distm[td], unit).cpu().numpy()
            pidx_3 = sd_idx(percent[2], distm[td], unit).cpu().numpy()
            pidx_4 = sd_idx(percent[3], distm[td], unit).cpu().numpy()
            pidx_5 = sd_idx(percent[4], distm[td], unit).cpu().numpy()

            ctn2 = 0
            ctn3 = 0
            ctn4 = 0
            ctn5 = 0

            lenpidx = np.array([len(pidx_1), len(pidx_2), len(pidx_3), len(pidx_4), len(pidx_5)])
            mask = lenpidx > 0
            minlen = np.sort(lenpidx[mask])[0]

            for b_idx, (path, _, _, idx) in enumerate(trainloader_div):
                idx = idx.cpu().numpy()
                (filepath, tempfilename) = os.path.split(path[0])

                if len(np.intersect1d(idx, pidx_1)) != 0:
                    if td < 10:
                        mymovefile(path[0], dir + '/t1/train/train_0' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    else:
                        mymovefile(path[0], dir + '/t1/train/train_' + str(td) + '/' + filepath.split('/')[-1] + '/')

                elif len(np.intersect1d(idx, pidx_2)) != 0 and ctn2 < minlen:
                    if td < 10:
                        mymovefile(path[0], dir + '/t2/train/train_0' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    else:
                        mymovefile(path[0], dir + '/t2/train/train_' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    ctn2 += 1

                elif len(np.intersect1d(idx, pidx_3)) != 0 and ctn3 < minlen:
                    if td < 10:
                        mymovefile(path[0], dir + '/t3/train/train_0' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    else:
                        mymovefile(path[0], dir + '/t3/train/train_' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    ctn3 += 1

                elif len(np.intersect1d(idx, pidx_4)) != 0 and ctn4 < minlen:
                    if td < 10:
                        mymovefile(path[0], dir + '/t4/train/train_0' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    else:
                        mymovefile(path[0], dir + '/t4/train/train_' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    ctn4 += 1

                elif len(np.intersect1d(idx, pidx_5)) != 0 and ctn5 < minlen:
                    if td < 10:
                        mymovefile(path[0], dir + '/t5/train/train_0' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    else:
                        mymovefile(path[0], dir + '/t5/train/train_' + str(td) + '/' + filepath.split('/')[-1] + '/')
                    ctn5 += 1