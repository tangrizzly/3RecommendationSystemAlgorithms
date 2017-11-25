#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 06/09/2017 9:54 PM
Modified on 25/11/2017 8:37 AM
@author: Tangrizzly
"""
import numpy as np
from datetime import datetime
from numpy.random import rand
# from numpy.random import uniform
from collections import Counter


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RnnBpr:
    def __init__(self, d, dataset, v, maxNo, lmd):
        self.d = d  # 隐层维度
        self.lmd = lmd  # lambda
        self.dataset = dataset
        self.v = v  # dataset里的items
        self.R = rand(maxNo + 1, d)  # 所有item的向量表示
        self.M = rand(self.d, self.d)  # uniform(-0.5, 0.5, (d, d))指定范围
        self.W = rand(self.d, self.d)

    def forward_propagation(self, x):
        K = len(x)
        h = np.zeros((K + 1, self.d))
        h[-1] = np.zeros(self.d)
        y = np.zeros(K).reshape((K, 1))  # 注意reshape函数的格式
        yi = np.zeros(K).reshape((K, 1))
        for k in np.arange(K):
            h[k] = sigmoid(np.dot(self.M, self.R[x[k]]) + np.dot(self.W, h[k - 1]))
            # 之前写的是点乘得到的结果
            y[k] = np.sum(h[k - 1] * self.R[x[k]])  # 对正样本的偏好值, 应该是pair-wise乘法之后加和
            i = int(np.random.choice(self.v, 1)[0])  # 负样本
            yi[k] = np.sum(h[k - 1] * self.R[i])  # 对负样本的偏好值
            # y_hat = np.argsort(np.dot(h[K-1], self.R.T))[-10:][::-1] # 将所有的评分进行排序之后取前十,计算量过大
            # 先得到前十之后进行排序
        scores = np.dot(h[K - 1], self.R.T)  # user对所有items的评分
        max_10 = np.argpartition(scores, -10)[-10:]  # 复杂度O(n),获得评分前十个的index
        y_hat = max_10[np.argsort(scores[max_10])][::-1]  # 复杂度O(nlogn),得到前十个得分最高的items_ids推荐给用户
        return h, y, yi, y_hat  # 得到前十个得分最大的对应的item_ids,并且最左边的是得分最大的

    def calculate_loss(self, lmd, x):
        h, y, yi, y_hat = self.forward_propagation(x)
        # l = np.sum(np.log(1 + np.exp(-y + yi)))           # ???
        l = np.sum(np.log(1 + np.exp(y - yi)))  # 对正样本偏好减对负样本偏好
        l += lmd / 2 * (np.square(np.linalg.norm(self.R))
                        + np.square(np.linalg.norm(self.W))
                        + np.square(np.linalg.norm(self.M))
                        )
        return l, y_hat

    def bptt(self, x, learning_rate):
        # 在此写的是BP,而非BPTT(汇总损失，仅更新一大步)
        K = len(x)
        h, y, yi, y_hat = self.forward_propagation(x)
        j = int(np.random.choice(self.v, 1)[0])
        dLdh = self.R[x[K - 1]] - self.R[j]
        dLdR = np.zeros(self.R.shape)
        for t in np.arange(K - 1)[::-1]:
            j = int(np.random.choice(self.v, 1)[0])
            Xij = np.dot(h[t], self.R[x[t + 1]].T) - np.dot(h[t], self.R[j].T)
            dLdR[x[t + 1]] = h[t]
            dLdR[j] = -h[t]
            df = np.multiply(h[t], (1 - h[t])).reshape(1, self.d)
            dLdM = np.dot(dLdh * df, self.R[x[t]].reshape((self.d, 1)))
            dLdW = np.dot(dLdh * df, h[t - 1].reshape((self.d, 1)))
            dLdh = np.dot(dLdh * df, self.W.T)
            self.M += learning_rate * (np.exp(-Xij) * sigmoid(Xij) * dLdM - self.lmd * self.M)
            self.W += learning_rate * (np.exp(-Xij) * sigmoid(Xij) * dLdW - self.lmd * self.W)
            self.R[x[t + 1]] += learning_rate * (
                np.exp(-Xij) * sigmoid(Xij) * dLdR[x[t + 1]] - self.lmd * self.R[x[t + 1]])
            self.R[j] += learning_rate * (np.exp(-Xij) * sigmoid(Xij) * dLdR[j] - self.lmd * self.R[j])


# def train(model, learning_rate=0.002):
#     """
#     :param model: RnnBpr
#     :param learning_rate: 一般采用0.1/0.01这种10倍间隔的，中间值可选用0.05这种。
#     :return: recall_rate
#     """
#     losses = []
#     recall = 0
#     for i in range(0, len(model.x)):
#         xi = model.x[0: i + 1]  # shape=(i+1, d)
#         loss, y_hat = model.calculate_loss(model.lmd, xi)
#         if i < len(model.x) - 1:
#             if model.x[i + 1] in y_hat:
#                 recall += 1
#         losses.append(loss)
#         time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         print "%s: Loss %f, recall %d"% (time, loss, recall)
#         model.bptt(xi, learning_rate)
#     recall_rate = recall / float(len(model.x) - 1)
#     return recall_rate

def train(model, learning_rate=0.01):
    """
    :param model: RnnBpr
    :param learning_rate: 一般采用0.1/0.01这种10倍间隔的，中间值可选用0.05这种。
    :return: recall_rate
    """
    example = 0
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print "%s: Loss after example=%d : %f" % (time, example, 0)
    for x in model.dataset:
        example += 1
        recall = 0
        for i in range(0, len(x)):
            xi = x[0: i + 1]  # shape=(i+1, d)
            loss, y_hat = model.calculate_loss(model.lmd, xi)
            if i < len(x) - 1:
                if x[i + 1] in y_hat:
                    recall += 1
            # time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # print "%s: Loss %f, recall %d"% (time, loss, recall)
            model.bptt(xi, learning_rate)
        recall_rate = recall / float(len(x) - 1)
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print "%s: Loss after example=%d : %f" % (time, example, loss)
        print "Recall rate: %f" % recall_rate


# 读取用户序列历史, n_user=737
with open('./data/user_cart.json', 'r') as f:
    data = f.readlines()
train_orig = []  # 嵌套列表,每个用户的序列=[int1, int2, ...]
temp = []
for line in data:
    # line = re.sub('[\[|\]|\s++]', '', line)
    # odom = line.split(',')
    odem = line[1:][:-2].replace(' ', '').split(',')
    if len(odem) < 10:
        continue
    # numbers_float = map(float, odom)
    # numbers = map(int, numbers_float)
    numbers = [int(float(i)) for i in odem]
    train_orig.append(numbers)
    temp.extend(numbers)

# 处理后,n_user=706
# remove items appearing less than 3 times
maxNo = 0
v = []
trainSet = []  # 嵌套列表，每个用户的序列=[int1, int2, ...]
# temp_df = pd.DataFrame(temp)
# lessThan3 = temp_df[0].value_counts()[temp_df[0].value_counts()<3].index[:]
items_count = dict(Counter(temp))
lessThan3 = [item for item, count in items_count.items() if count < 3]
for line in train_orig:
    cleaned = [x for x in line if x not in lessThan3]
    maxNo = np.max([maxNo, np.max(cleaned)])
    v = np.append(v, np.unique(cleaned))
    if len(cleaned) >= 10:
        trainSet.append(cleaned)

np.random.seed(1)
recall = []
# model可以只建1次，一次性初始化所有的items表达
model = RnnBpr(10, trainSet, v, maxNo, 0.01)
train(model)
# np.savetxt('rnn_bpr_result', np.asarray(recall), fmt='%f')
