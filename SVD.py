#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 06/09/2017 9:50 AM

@author: Tangrizzly
"""
# best result:
# rse_train = 57372.4818287
# rse_test = 21230.4729627
# more loops will lead to overfitting

import numpy as np

test_orgi = np.loadtxt("./data/test.txt")
train_orgi = np.loadtxt("./data/train.txt")

u = np.unique(train_orgi[:, 0]).shape[0] + 1
# i = np.unique(train_orgi[:, 0]).shape[0]
i = int(np.max(train_orgi[:, 1])) + 1
f = 20
gm = 0.005
lmd = 0.02
avg = np.average(train_orgi[:, 2])

# initialization
qi = np.random.rand(f, i)
pu = np.random.rand(f, u)
bi = np.zeros([i, 1])
bu = np.zeros([1, u])
# r_hat = avg + bi + bu + np.dot(qi.T, pu)

for i in range(0, 19):
    print i
    for rui in train_orgi:
        a = int(rui[0])
        b = int(rui[1])
        rui_hat = avg + bi[b][0] + bu[0][a] + np.dot(qi[:, b].T, pu[:, a])
        eui = rui[2] - rui_hat
        bu[0][a] += gm * (eui - lmd * bu[0][a])
        bi[b][0] += gm * (eui - lmd * bi[b][0])
        qi[:, b] += gm * (eui * pu[:, a] - lmd * qi[:, b])
        pu[:, a] += gm * (eui * qi[:, b] - lmd * pu[:, a])
    rse_train = 0
    for rui in train_orgi:
        a = int(rui[0])
        b = int(rui[1])
        rui_hat = avg + bi[b][0] + bu[0][a] + np.dot(qi[:, b].T, pu[:, a])
        rse_train += np.square(rui[2]-rui_hat) + lmd*(np.square(bi[b][0])+np.square(bu[0][a])+np.square(np.linalg.norm(qi[:, b]))+np.square(np.linalg.norm(pu[:, a])))
    print rse_train
    rse_test = 0
    for rui in test_orgi:
        a = int(rui[0])
        b = int(rui[1])
        if b >= 1680:
            rui_hat = avg
            rse_test += 0
        else:
            rui_hat = avg + bi[b][0] + bu[0][a] + np.dot(qi[:, b].T, pu[:, a])
            rse_test += np.square(rui[2]-rui_hat) + lmd*(np.square(bi[b][0])+np.square(np.linalg.norm(qi[:, b]))+np.square(np.linalg.norm(pu[:, a])))
    print rse_test

for rui in test_orgi:
    a = int(rui[0])
    b = int(rui[1])
    if b>=1680:
        rui_hat=avg
    else:
        rui_hat = avg + bi[b][0] + bu[0][a] + np.dot(qi[:, b].T, pu[:, a])
    rui[3] = rui_hat
    print rui

np.savetxt('result', test_orgi, fmt='%.2f')



