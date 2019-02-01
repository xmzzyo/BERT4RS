import gzip
import os

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

dataset_name = 'Digital_Music_5'

vocab = set()
for l in tqdm(parse('reviews_' + dataset_name + '.json.gz'), total=line):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    countU[rev] += 1
    countP[asin] += 1
    vocab.add(asin)
with open(os.path.join(dataset_name, 'vocab.txt'), 'w') as f:
    f.write("[UNK]\n[CLS]\n[SEP]\n[MASK]\n[PAD]\n")
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
    if countU[rev] < 5 or countP[asin] < 5:
        continue
    if rev not in list(User.keys()):
        User[rev] = []
    User[rev].append([time, asin])

# sort reviews in User according to time
print('start sort')
for userid in list(User.keys()):
    User[userid].sort(key=lambda x: x[0])

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

tim_itv = 800 * 3600
max_length = 60
min_length = 5
seq_lens = []
# 之后递归重写
with open(os.path.join(dataset_name, 'data_' + dataset_name + '.txt'), 'w') as f:
    for user in tqdm(list(User.values())):
        time_interval = [u[0] for u in user]
        time_interval = [time_interval[i] - time_interval[i - 1] for i in range(1, len(time_interval))]
        time_interval.insert(0, 0)
        assert len(time_interval) == len(user)
        seq = [u[1] for u in user]
        split_idx = [0, len(time_interval)]
        has_max_length = True
        split_idx.sort()
        i = 0
        while i < len(split_idx) - 1:
            split_idx.sort()
            start = split_idx[i]
            end = split_idx[i + 1]
            len_tmp = end - start
            if len_tmp > max_length:
                max_idx = np.argmax(time_interval[start:end])
                # tim_tmp = time_interval[start: end]
                # tim_tmp.sort(reverse=True)
                # for j in range(len_tmp):
                #     if max_idx < min_length or len_tmp - max_idx < min_length:
                #         max_idx = time_interval.index(tim_tmp[j]) - start
                #     else:
                #         break
                split_idx.append(max_idx + split_idx[i])
                time_interval[max_idx + split_idx[i]] = -1
                i -= 1
            i += 1
        split_idx.sort()
        seqs = []
        for i in range(len(split_idx) - 1):
            seqs.append(seq[split_idx[i]:split_idx[i + 1]])
        for seq in seqs:
            seq_lens.append(len(seq))
            f.write(' '.join(seq) + '\n')
        f.write('\n')

print(np.amax(seq_lens), np.amin(seq_lens), np.mean(seq_lens), np.median(seq_lens), np.std(seq_lens))
action_num = 0
with open(os.path.join(dataset_name, 'data_' + dataset_name + '.txt'), 'r') as f:
    for l in f.readlines():
        if len(l) > 1:
            action_num += len(l.strip().split())
print(action_num)

# user item
# action

# book
# 603668    367982
# 8898041

# digital music
# 5541  3568
# 64706
