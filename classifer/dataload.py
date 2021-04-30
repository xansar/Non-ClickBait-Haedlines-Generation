import numpy as np
import random
import torch
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# 读取数据
def readFile(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
    return content


# 读取标题党词汇表
def read_clickbait_dic(filename="../Knowledge/clickbait_dic.txt"):
    with open(filename, encoding="utf-8") as f:
        clickbait_dic = f.readlines()
    return clickbait_dic


# 加载数据
def loadData(dataname):
    ############### 加载训练集 ###############
    True_Train_titles, True_Train_contents = (
        readFile("../Data/{}/{}_True_Train_titles.txt".format(dataname, dataname)),
        readFile("../Data/{}/{}_True_Train_contents.txt".format(dataname, dataname)),
    )
    True_Train = list(zip(True_Train_titles, True_Train_contents))

    False_Train_titles, False_Train_contents = (
        readFile("../Data/{}/{}_False_Train_titles.txt".format(dataname, dataname)),
        readFile("../Data/{}/{}_False_Train_contents.txt".format(dataname, dataname)),
    )
    False_Train = list(zip(False_Train_titles, False_Train_contents))
    ############### 加载测试集 ###############
    True_Test_titles, True_Test_contents = (
        readFile("../Data/{}/{}_True_Test_titles.txt".format(dataname, dataname)),
        readFile("../Data/{}/{}_True_Test_contents.txt".format(dataname, dataname)),
    )
    True_Test = list(zip(True_Test_titles, True_Test_contents))

    False_Test_titles, False_Test_contents = (
        readFile("../Data/{}/{}_False_Test_titles.txt".format(dataname, dataname)),
        readFile("../Data/{}/{}_False_Test_contents.txt".format(dataname, dataname)),
    )
    False_Test = list(zip(False_Test_titles, False_Test_contents))

    ############### 设定训练集标签 ###############
    True_Train_targets = np.ones([len(True_Train)])
    False_Train_targets = np.zeros([len(False_Train)])

    Train_targets = np.concatenate(
        (True_Train_targets, False_Train_targets), axis=0
    ).reshape(-1, 1)
    Train_targets = torch.tensor(Train_targets)

    ############### 设定测试集标签 ###############
    True_Test_targets = np.ones([len(True_Test)])
    False_Test_targets = np.zeros([len(False_Test)])
    Test_targets = np.concatenate(
        (True_Test_targets, False_Test_targets), axis=0
    ).reshape(-1, 1)
    Test_targets = torch.tensor(Test_targets)
    # print(
    #     "True_Train Length: {}, False_Train Length: {}, True_Test Length: {}, False_Test Length: {}".format(
    #         len(True_Train), len(False_Train), len(True_Test), len(False_Test)
    #     )
    # )

    return True_Train, False_Train, True_Test, False_Test, Train_targets, Test_targets


# 将每一句转成数字 （加上首位两个标识，大于limit_size做截断，小于limit_size做 Padding）
def convert_text_to_token(tokenizer, sentence, limit_size, clickbait_dic=None):
    title, content = sentence
    include_clickbait = ""
    if clickbait_dic is not None:
        for words in clickbait_dic:
            words_list = words.split()
            # print(words_list)
            tag = True
            for word in words_list:
                if word not in title:
                    tag = False
            # 该“标题党”热词包含在标题中
            if tag is True:
                include_clickbait += words + "、"
        input_tokens = (
            ["[CLS]"]
            + tokenizer.tokenize(title)
            + ["[SEP]"]
            + tokenizer.tokenize(include_clickbait)
            + ["[SEP]"]
            + tokenizer.tokenize(content)
            + ["[SEP]"]
        )
    else:
        input_tokens = (
            ["[CLS]"]
            + tokenizer.tokenize(title)
            + ["[SEP]"]
            + tokenizer.tokenize(content)
            + ["[SEP]"]
        )

    if len(input_tokens) > limit_size - 1:
        input_tokens = input_tokens[: limit_size - 1] + ["[SEP]"]
    # print(input_tokens)
    tokens = tokenizer.convert_tokens_to_ids(input_tokens)

    # tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    if len(tokens) < limit_size:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size - len(tokens)))
    return tokens


# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:  # [10000, 128]
        seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
        atten_masks.append(seq_mask)
    return atten_masks


# 处理加载好的数据
def solve_data(
    tokenizer,
    dataname="ClickBait",
    limit_size=256,
    BATCH_SIZE=32,
    using_clickbait_dic=False,
):
    # 读取数据
    (
        True_Train,
        False_Train,
        True_Test,
        False_Test,
        Train_targets,
        Test_targets,
    ) = loadData(dataname)

    if using_clickbait_dic is True:
        # 加载标题党新闻词汇表
        clickbait_dic = read_clickbait_dic(
            filename="../Knowledge/{}_vob.txt".format(dataname)
        )
        clickbait_dic = [_.strip() for _ in clickbait_dic]
        print("###### Using Clickbait Dic ######")
    else:
        clickbait_dic = None
    # 观察数据
    # print(clickbait_dic)
    # print(True_Train[2])
    # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(True_Train[2])))
    # print(tokenizer.encode(True_Train[2]))
    # print(tokenizer.convert_ids_to_tokens(tokenizer.encode(True_Train[2])))

    # Train
    True_Train_ids = [
        convert_text_to_token(tokenizer, sen, limit_size, clickbait_dic)
        for sen in True_Train
    ]
    False_Train_ids = [
        convert_text_to_token(tokenizer, sen, limit_size, clickbait_dic)
        for sen in False_Train
    ]
    Train_ids = True_Train_ids + False_Train_ids
    Train_tokens = torch.tensor(Train_ids)

    Train_tokens, Train_targets = shuffle(Train_tokens, Train_targets)
    print("Train:", Train_tokens.shape)  # torch.Size([ , 128])

    # Test
    True_Test_ids = [
        convert_text_to_token(tokenizer, sen, limit_size, clickbait_dic)
        for sen in True_Test
    ]
    False_Test_ids = [
        convert_text_to_token(tokenizer, sen, limit_size, clickbait_dic)
        for sen in False_Test
    ]
    Test_ids = True_Test_ids + False_Test_ids
    Test_tokens = torch.tensor(Test_ids)

    Test_tokens, Test_targets = shuffle(Test_tokens, Test_targets)
    print("Test:", Test_tokens.shape)  # torch.Size([ , 128])

    # 加Mask
    # Train
    Train_atten_masks = attention_masks(Train_tokens)
    Train_attention_tokens = torch.tensor(Train_atten_masks)
    print("Train:", Train_attention_tokens.shape)  # torch.Size([ , 128])

    # Test
    Test_atten_masks = attention_masks(Test_tokens)
    Test_attention_tokens = torch.tensor(Test_atten_masks)
    print("Test:", Test_attention_tokens.shape)  # torch.Size([ , 128])

    # 封装成DataLoader
    train_data = TensorDataset(Train_tokens, Train_attention_tokens, Train_targets)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=BATCH_SIZE
    )

    test_data = TensorDataset(Test_tokens, Test_attention_tokens, Test_targets)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, test_dataloader


def solve_test_data(
    tokenizer,
    dataname="appraise",
    limit_size=256,
    BATCH_SIZE=16,
    using_clickbait_dic=False,
):
    print("###### {} ######".format(dataname))
    appraise_titles, appraise_contents = (
        readFile("../Data/Test/{}/{}_titles.txt".format(dataname, dataname)),
        readFile("../Data/Test/{}/{}_contents.txt".format(dataname, dataname)),
    )
    appraise = list(zip(appraise_titles, appraise_contents))

    if using_clickbait_dic is True:
        # 加载标题党新闻词汇表
        clickbait_dic = read_clickbait_dic(
            filename="../Knowledge/{}_vob.txt".format(dataname)
        )
        clickbait_dic = [_.strip() for _ in clickbait_dic]
        print("###### Using Clickbait Dic ######")
    else:
        clickbait_dic = None

    appraise_ids = [
        convert_text_to_token(tokenizer, sen, limit_size, clickbait_dic)
        for sen in appraise
    ]

    appraise_tokens = torch.tensor(appraise_ids)
    print("appraise:", appraise_tokens.shape)
    appraise_atten_masks = attention_masks(appraise_tokens)
    appraise_attention_tokens = torch.tensor(appraise_atten_masks)
    # print("appraise:", appraise_attention_tokens.shape)  # torch.Size([ , 128])

    # 封装成DataLoader
    appraise_data = TensorDataset(appraise_tokens, appraise_attention_tokens)
    appraise_sampler = RandomSampler(appraise_data)
    appraise_dataloader = DataLoader(
        appraise_data, sampler=appraise_sampler, batch_size=BATCH_SIZE
    )

    return appraise_dataloader


# from transformers import BertTokenizer

# if __name__ == "__main__":
#     tokenizer = BertTokenizer.from_pretrained(
#         "bert-base-chinese", cache_dir="../sample_data/"
#     )
#     solve_data(tokenizer, dataname="ClickBait")
