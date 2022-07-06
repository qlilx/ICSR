from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from sklearn import metrics


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def pre_tar_(model, td, ncl):
    N = len(td.dataset)
    targets = torch.zeros(N, dtype=torch.int64).cuda()
    onk = torch.zeros(N, ncl).cuda()
    model.eval()
    for batch_idx, batch in enumerate(td):
        data = batch['image'].cuda()
        tgs = batch['target'].cuda()
        idx = batch['meta']['index']
        onk[idx, :] = model(data)[0].detach()
        targets[idx] = tgs
    predictions = torch.argmax(onk, 1)

    return predictions, targets


def pre_tar(model, td, ncl):
    N = len(td.dataset)
    targets = torch.zeros(N, dtype=torch.int64).cuda()
    onk = torch.zeros(N, ncl).cuda()
    model.eval()
    for batch_idx, (data, tgs, idx) in enumerate(td):
        data = data.cuda()
        tgs = tgs.cuda()
        onk[idx, :] = model(data).detach()
        targets[idx] = tgs
    predictions = torch.argmax(onk, 1)

    return predictions, targets


def pre_tar_spice(model, td, ncl):
    N = len(td.dataset)
    targets = torch.zeros(N, dtype=torch.int64).cuda()
    onk = torch.zeros(N, ncl).cuda()
    model.eval()
    for batch_idx, (_, data, tgs, idx) in enumerate(td):
        data = data.cuda()
        tgs = tgs.cuda()
        onk[idx, :] = model(data).detach()
        targets[idx] = tgs
    predictions = torch.argmax(onk, 1)

    return predictions, targets


import numpy as np

nlb = 10


def acc(model, td, ncl):
    lbs, tlb = pre_tar(model, td, ncl)
    N = len(td.dataset)
    dnp = np.zeros((ncl, nlb), dtype=np.int32)
    for ii in range(N):
        dnp[lbs[ii], tlb[ii]] += 1
    dis = np.zeros(ncl)
    max = 0
    for i in range(ncl):
        print(i, ':  ', np.sum(dnp[i]), '  ', dnp[i], '  ', np.argmax(dnp[i]))
        dis[i] = np.sum(dnp[i])
        max = max + np.max(dnp[i])

    print(np.sum(dnp))
    print(np.std(dis))
    print(max)
    print(max / N)

    return max / N


def humgarian_acc(predictions, targets, num_classes):
    num_elems = len(targets)
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    print('acc: ', acc, ' mni: ', nmi, ' ari: ', ari)

    return acc