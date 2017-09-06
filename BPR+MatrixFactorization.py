#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 06/09/2017 6:05 PM

@author: Tangrizzly
"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update(w, hi, hj, learning_rate, regularization):
    xi = np.dot(w, hi.T)
    xj = np.dot(w, hj.T)
    xij = xi - xj
    w += learning_rate * (np.exp(-xij) * sigmoid(xij) * (hi - hj) + w * regularization)
    hi += learning_rate * (np.exp(-xij) * sigmoid(xij) * (w) + hi * regularization)
    hj += learning_rate * (np.exp(-xij) * sigmoid(xij) * (-w) + hj * regularization)


def rse_train():
    rse_train = 0
    for rui in train_orgi:
        utrain = int(rui[0])
        itrain = int(rui[1])
        rui_hat = np.dot(W[utrain, :], H[itrain, :].T)
        rse_train += np.square(rui[2] - rui_hat) + lmd * (
        np.square(np.linalg.norm(W[utrain, :])) + np.square(np.linalg.norm(H[itrain, :].T)))
    return rse_train


def rse_test():
    rse_test = 0
    for rui in test_orgi:
        utest = int(rui[0])
        itest = int(rui[1])
        if itest>=1680:
            rse_test += 0
        else:
            rui_hat = np.dot(W[utest, :], H[itest, :].T)
            rse_test += np.square(rui[2] - rui_hat) + lmd * (
                np.square(np.linalg.norm(W[utest, :])) + np.square(np.linalg.norm(H[itest, :].T)))
    return rse_test

test_orgi = np.loadtxt("./data/test.txt")
train_orgi = np.loadtxt("./data/train.txt")

us = np.unique(train_orgi[:, 0]).shape[0]+1
it = int(np.max(train_orgi[:, 1])+1)
avg = np.average(train_orgi[:, 2])

# initialization
f = 20
gm = 0.005
lmd = 0.02
W = np.random.rand(us, f)
H = np.random.rand(it, f)
for b in range(0, 50):
    for a in range(0, 1000):
        u = np.random.choice(train_orgi[:, 0], 1)[0]
        [i, j] = np.random.choice(train_orgi[:, 1], 2)
        update(W[int(u), :], H[int(i), :], H[int(j), :], gm, lmd)
    print rse_train()
    print rse_test()


# best result: 34*1000 times bootstrap sampling
# train 358435.052103
# test 87530.4892336
