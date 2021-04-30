import torch
import time
import datetime


# 准确率
def binary_acc(preds, labels):  # preds.shape = [16, 2] labels.shape = [16, 1]
    # torch.max: [0]为最大值, [1]为最大值索引
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    acc = correct.sum().item() / len(correct)
    return acc


# 精度、召回率、F1
def binary_eva(preds, targets):  # preds.shape = [16, 2] labels.shape = [16, 1]
    precision, recall, F1 = 0, 0, 0

    TP = ((preds == 1) & (targets == 1)).cpu().sum()
    TN = ((preds == 0) & (targets == 0)).cpu().sum()
    FN = ((preds == 0) & (targets == 1)).cpu().sum()
    FP = ((preds == 1) & (targets == 0)).cpu().sum()

    if TP + FP != 0:
        precision = TP / (TP + FP)

    if TP + FN != 0:
        recall = TP / (TP + FN)

    if recall != 0 and precision != 0:
        F1 = 2 * recall * precision / (recall + precision)
    return precision, recall, F1


# 评估时间
def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(
        datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间