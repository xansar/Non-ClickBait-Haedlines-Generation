import numpy as np
import random
import torch
import os
import time
import matplotlib.pylab as plt

from torch.nn.utils import clip_grad_norm_
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    AlbertTokenizer,
    BertModel,
)
from transformers import get_linear_schedule_with_warmup
import dataload
import estimate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 超参数
SEED = 123
BATCH_SIZE = 32
epochs = 6
Limit_size = 256
learning_rate = 2e-5
weight_decay = 1e-2
epsilon = 1e-8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model_names = [
    "bert-base-chinese",
    "hfl/chinese-bert-wwm",
    "hfl/chinese-bert-wwm-ext",
    "../bert-base-uncased",
]
model_name = model_names[2]
cache_dir = "./sample_data/"
DataName = "ClickBait"
USING_CLICKBAIT_DIC = True

print("Task: {}".format(DataName))
print("Using pretrain Model: {}".format(model_name))

# 训练
def train(model, device, train_dataloader, optimizer, scheduler):
    t0 = time.time()
    avg_loss, avg_acc = [], []
    preds, targets = torch.tensor([]).to(device), torch.tensor([]).to(device)

    model.train()
    for step, batch in enumerate(train_dataloader):

        b_input_ids, b_input_mask, b_labels = (
            batch[0].long().to(device),
            batch[1].long().to(device),
            batch[2].long().to(device),
        )

        output = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss, logits = output[0], output[1]  # loss: 损失, logits: predict

        avg_loss.append(loss.item())

        acc = estimate.binary_acc(logits, b_labels)  # (predict, label)
        avg_acc.append(acc)

        preds = torch.cat((preds, torch.max(logits, dim=1)[1]))
        targets = torch.cat((targets, b_labels.flatten()))

        optimizer.zero_grad()  # 清空optimizer的梯度
        loss.backward()  # 计算梯度
        clip_grad_norm_(model.parameters(), 1.0)  # 大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新learning rate

        # 每隔64个batch 输出一下所用时间
        if step % 64 == 0 and not step == 0:
            elapsed = estimate.format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    precision, recall, F1 = estimate.binary_eva(preds, targets)
    return avg_loss, avg_acc, precision, recall, F1


# 评估
def evaluate(model, device, test_dataloader):

    avg_acc = []
    preds, targets = torch.tensor([]).to(device), torch.tensor([]).to(device)

    model.eval()  # 表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = (
                batch[0].long().to(device),
                batch[1].long().to(device),
                batch[2].long().to(device),
            )

            output = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

            acc = estimate.binary_acc(output[0], b_labels)
            avg_acc.append(acc)

            preds = torch.cat((preds, torch.max(output[0], dim=1)[1]))
            targets = torch.cat((targets, b_labels.flatten()))

    precision, recall, F1 = estimate.binary_eva(preds, targets)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc, precision, recall, F1


def appraise(modelpath, dataname, device="cuda:0"):

    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(modelpath, cache_dir=cache_dir)
    model = BertForSequenceClassification.from_pretrained(
        modelpath, num_labels=2
    )  # num_labels表示2个分类,好评和差评
    model.to(device)

    appraise_dataloader = dataload.solve_test_data(tokenizer, dataname)
    appraise_list = np.array([])

    # model.eval()  # 表示进入测试模式
    with torch.no_grad():
        for batch in appraise_dataloader:
            b_input_ids, b_input_mask = (
                batch[0].long().to(device),
                batch[1].long().to(device),
            )

            output = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )
            appraise = torch.max(output[0].to("cpu"), dim=1)[1].numpy()
            appraise_list = np.concatenate((appraise_list, appraise), axis=0)
    return appraise_list


def start_appraise():
    DATANAME = "solved_appraise2"
    print("###### {} ######".format(DATANAME))
    result = appraise("./save_model", dataname=DATANAME)
    print(result)
    with open("./result/{}.txt".format(DATANAME), "w", encoding="utf-8") as f:
        for r in result:
            f.write("{}\n".format(r))
    print(
        "Len:{} Num:{} Percentage:{}".format(
            len(result), result.sum(), result.sum() / len(result)
        )
    )


def main():
    # 创建模型
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2, mirror="tuna"
    )  # num_labels表示2个分类,好评和差评
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载数据
    if model_name == "../bert-base-uncased":
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name, mirror="tuna")

    train_dataloader, test_dataloader = dataload.solve_data(
        tokenizer,
        dataname=DataName,
        limit_size=Limit_size,
        BATCH_SIZE=BATCH_SIZE,
        using_clickbait_dic=USING_CLICKBAIT_DIC,
    )

    # training steps 的数量: [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs

    # 定义优化方法
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon)

    # 设计 learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    records_Train = {}
    records_Test = {}
    best_epoch = -1
    best_score = -1
    for epoch in range(epochs):
        train_loss, train_acc, train_precision, train_recall, train_F1 = train(
            model, device, train_dataloader, optimizer, scheduler
        )
        print(
            "epoch={}, 训练准确率={}, 损失={}, 精度={}, 召回率={}, F1={}".format(
                epoch, train_acc, train_loss, train_precision, train_recall, train_F1
            )
        )

        test_acc, test_precision, test_recall, test_F1 = evaluate(
            model, device, test_dataloader
        )
        print(
            "epoch={}, 测试准确率={}, 精度={}, 召回率={}, F1={}".format(
                epoch, test_acc, test_precision, test_recall, test_F1
            )
        )

        records_Train["Epoch" + str(epoch)] = [
            train_acc,
            train_loss,
            train_precision,
            train_recall,
            train_F1,
        ]

        records_Test["Epoch" + str(epoch)] = [
            test_acc,
            test_precision,
            test_recall,
            test_F1,
        ]

        if test_F1 > best_score:
            best_score = test_F1
            best_epoch = epoch

        # if epoch == 2:
        #     print("###### Save Model ######")
        #     model.save_pretrained("./save_model")
        #     tokenizer.save_pretrained("./save_model")

    print("###### Finished ######")
    train_acc, train_loss, train_precision, train_recall, train_F1 = records_Train[
        "Epoch" + str(best_epoch)
    ]
    test_acc, test_precision, test_recall, test_F1 = records_Test[
        "Epoch" + str(best_epoch)
    ]
    print(
        "best_epoch={}, 训练准确率={}, 损失={}, 精度={}, 召回率={}, F1={}".format(
            best_epoch, train_acc, train_loss, train_precision, train_recall, train_F1
        )
    )
    print(
        "best_epoch={}, 测试准确率={}, 精度={}, 召回率={}, F1={}".format(
            best_epoch, test_acc, test_precision, test_recall, test_F1
        )
    )


if __name__ == "__main__":
    # execute only if run as a script
    main()
    # start_appraise()
