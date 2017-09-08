#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 06/09/2017 9:54 PM

@author: Tangrizzly
"""
import re
import numpy as np
from datetime import datetime
import sys
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RnnBpr:
    def __init__(self, d, x, lmd):
        self.d = d
        self.x = x
        self.lmd = lmd
        self.v = np.unique(x)
        self.R = np.random.rand(np.max(x)+1, d)
        self.M = np.random.rand(d, d)
        self.W = np.random.rand(d, d)

    def forward_propagation(self):
        K = len(self.x)
        h = np.zeros((K + 1, self.d))
        h[-1] = np.zeros(self.d)
        y = np.zeros(K).reshape(K, 1)
        yi = np.zeros(K).reshape(K, 1)
        recall = 0
        for k in np.arange(K):
            h[k] = sigmoid(np.dot(self.R[self.x[k]].reshape(1, self.d), self.M) + np.dot(h[k - 1].reshape(1, self.d), self.W))
            # y[k] = np.dot(np.dot(h[k], self.W), np.dot(self.R[x[k]], self.M).T)
            y[k] = np.dot(h[k-1].reshape(1, self.d), self.R[self.x[k]].reshape(self.d, 1))
            i = np.random.choice(self.v, 1)[0]
            # yi[k] = np.dot(np.dot(h[k], self.W), np.dot(r[i], self.M).T)
            yi[k] = np.dot(h[k-1].reshape(1, self.d), self.R[i].reshape(self.d, 1))
            y_hat = np.argsort(np.dot(h[k - 1], self.R.T))[-10:][::-1]
            if self.x[k] in y_hat:
                recall += 1
        recall_rate = recall/float(len(self.x))
        return h, y, yi, recall_rate

    def calculate_loss(self, lmd):
        h, y, yi, recall_rate = self.forward_propagation()
        l = np.sum(np.log(1 + np.exp(-y + yi)))
        l += lmd / 2 * (np.square(np.linalg.norm(self.R))
                        + np.square(np.linalg.norm(self.W))
                        + np.square(np.linalg.norm(self.M))
                        )
        return l, recall_rate

    def bptt(self, learning_rate):
        K = len(self.x)
        h, y, yi, recall_rate = self.forward_propagation()
        j = np.random.choice(self.v, 1)[0]
        dLdh = self.R[self.x[K-1]] - self.R[j]
        dLdR = np.zeros(self.R.shape)
        for t in np.arange(K-1)[::-1]:
            j = np.random.choice(self.v, 1)[0]
            Xij = np.dot(h[t], self.R[self.x[t+1]].T) - np.dot(h[t], self.R[j].T)
            dLdR[self.x[t+1]] = h[t]
            dLdR[j] = -h[t]
            df = np.multiply(h[t], (1 - h[t])).reshape(1, self.d)
            dLdM = np.dot(self.R[self.x[t]].reshape(self.d, 1), dLdh * df)
            dLdW = np.dot(h[t-1].reshape(self.d, 1), dLdh * df)
            dLdh = np.dot(dLdh * df, self.W.T)
            self.M += learning_rate * (np.exp(-Xij)*sigmoid(Xij)*dLdM-self.lmd*self.M)
            self.W += learning_rate * (np.exp(-Xij)*sigmoid(Xij)*dLdW-self.lmd*self.W)
            self.R[self.x[t+1]] += learning_rate * (np.exp(-Xij)*sigmoid(Xij)*dLdR[self.x[t+1]]-self.lmd*self.R[self.x[t+1]])
            self.R[j] += learning_rate * (np.exp(-Xij)*sigmoid(Xij)*dLdR[j]-self.lmd*self.R[j])


def train(model, learning_rate=0.002, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    max_recall_rate = 0.0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if epoch % evaluate_loss_after == 0:
            loss, recall_rate = model.calculate_loss(model.lmd)
            max_recall_rate = np.max([max_recall_rate, recall_rate])
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d : %f" % (time, num_examples_seen, loss)
            print "Max recall rate: %f" % max_recall_rate
            # Adjust the learning rate if loss increases
            # if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
            #    learning_rate = learning_rate * 0.5
            #    print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            model.bptt(learning_rate)
            num_examples_seen += 1
    return max_recall_rate


with open('./data/user_cart.json', 'r') as f:
    data = f.readlines()
train_orig = []
temp = []
for line in data:
    line = re.sub('[\[|\]|\s++]', '', line)
    odom = line.split(',')
    if len(odom) < 10:
        continue
    numbers_float = map(float, odom)
    numbers = map(int, numbers_float)
    train_orig.append(numbers)
    temp.extend(numbers)

# remove items appearing less than 3 times
trainSet = []
testSet = []
temp_df = pd.DataFrame(temp)
lessThan3 = temp_df[0].value_counts()[temp_df[0].value_counts()<3].index[:]
for line in train_orig:
    cleaned = [x for x in line if x not in lessThan3]
    if len(cleaned) >= 10:
        trainSet.append(cleaned)
    # split train set and test set
    # if len(cleaned) >= 10:
    #    trainSet.append(cleaned[0: int(len(cleaned) * 0.8)-1])
    #    testSet.append(cleaned[int(len(cleaned) * 0.8): -1])

np.random.seed(1)
recall = []
for rui in np.arange(len(trainSet)):
    model = RnnBpr(10, trainSet[rui], 0.01)
    recall.append(train(model, evaluate_loss_after=1))
    print rui
np.savetxt('rnn_bpr_result', np.asarray(recall), fmt='%f')