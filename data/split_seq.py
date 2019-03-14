# -*- coding: utf-8 -*-
import gzip
import os
import random

import numpy as np
from collections import defaultdict

from tqdm import tqdm


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

dataset_name = 'Movies_and_TV_5'
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

vocab = set()
for l in tqdm(parse('reviews_' + dataset_name + '.json.gz'), total=line):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    countU[rev] += 1
    countP[asin] += 1
    vocab.add(asin)
with open(os.path.join(dataset_name, 'vocab.txt'), 'w') as f:
    f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    for asin in vocab:
        f.write(asin + "\n")

print(len(countU), len(countP))
line = 0
User = dict()
for l in tqdm(parse('reviews_' + dataset_name + '.json.gz'), total=line):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    if countU[rev] < 10 or countP[asin] < 10:
        continue
    if rev not in list(User.keys()):
        User[rev] = []
    User[rev].append([time, asin])

# sort reviews in User according to time
print('start sort')
for userid in list(User.keys()):
    User[userid].sort(key=lambda x: x[0])

with open(os.path.join(dataset_name, 'data_' + dataset_name + '_10_core.txt'), 'w') as f:
    for userid in list(User.keys()):
        f.write(" ".join([str(u[1]) for u in User[userid]]) + "\n")

seq_lens = np.asarray([len(u) for u in User.values()])
print(np.amax(seq_lens), np.amin(seq_lens), np.mean(seq_lens), np.median(seq_lens), np.std(seq_lens))

interval_res = []
for u in User.values():
    time_interval = [i[0] for i in u]
    time_interval = [(time_interval[i] - time_interval[i - 1]) / 3600 for i in range(1, len(time_interval))]
    time_interval.insert(0, 0)
    interval_res.append(
        [np.amax(time_interval), np.amin(time_interval), np.mean(time_interval), np.median(time_interval),
         np.std(time_interval)])
print(np.amax(interval_res, axis=0), np.mean(interval_res, axis=0), np.median(interval_res, axis=0))

actions = [u[0] for s in User.keys() for u in s]


def train_test_split(data_pairs, test_ratio, shuffle=False):
    n_total = len(data_pairs)
    offset = int(n_total * test_ratio)
    if n_total == 0 or offset < 1:
        return [], data_pairs
    if shuffle:
        random.shuffle(data_pairs)
    test = data_pairs[:offset]
    train = data_pairs[offset:]
    return train, test


pretrain_ratio = 0.5
pretrain, test = train_test_split(list(User.values()), pretrain_ratio, True)
print(len(pretrain), len(test))

seqs = []
seq_lens = []
tim_itv = 800 * 3600
max_length = 62
min_length = 10
with open(os.path.join(dataset_name, 'pretrain_' + dataset_name + '.txt'), 'w') as f:
    for user in tqdm(pretrain):
        seq = []
        start = len(user)
        end = len(user)
        action = [u[1] for u in user]
        # time_interval = [u[0] for u in user]
        # for t in range(len(time_interval) - 1, 0, -1):
        #     if time_interval[t] - time_interval[t - 1] > tim_itv:
        #         start = t
        #         if end - start > 10:
        #             seq.append(action[start:end])
        #             end = start
        nbatch = len(action) // max_length
        for i in range(nbatch):
            seq.append(action[len(action) - (i + 1) * max_length:len(action) - i * max_length])
        if len(action) > nbatch * max_length + 10:
            seq.append(action[0:len(action) - nbatch * max_length])
        seq.reverse()
        seqs.append(seq)
        for s in seq:
            seq_lens.append(len(s))
            f.write(' '.join(s) + '\n')
        if len(seq) > 0:
            f.write('\n')

print(np.sum(seq_lens), np.amax(seq_lens), np.amin(seq_lens), np.mean(seq_lens), np.median(seq_lens), np.std(seq_lens))
seq_lens = []
with open(os.path.join(dataset_name, 'test_' + dataset_name + '.txt'), 'w') as f:
    for user in test:
        seq_lens.append(len(user))
        f.write(' '.join([u[1] for u in user]) + '\n')

print(np.sum(seq_lens), np.amax(seq_lens), np.amin(seq_lens), np.mean(seq_lens), np.median(seq_lens), np.std(seq_lens))

# user item
# action

# book
# 603668    367982
# 8898041

# digital music
# 5541  3568
# 64706
