import sys
import copy
import random
import numpy as np
from collections import defaultdict

from tqdm import tqdm


def get_all_dataset(fname):
    User = defaultdict(list)
    with open('../data/%s.txt' % fname, 'r') as f:
        for line, u in enumerate(f.readlines()):
            i = line.rstrip().split(' ')
            User[u] = i
    return User


def data_partition(fname, seq_len, tokenizer):
    user_train = {}
    user_valid = {}
    user_test = {}

    seqs = []
    with open(fname, "r", encoding="utf8") as f:
        for line in tqdm(f.readlines(), desc="Loading Dataset"):
            line = line.strip()
            tokens = line.rstrip().split(' ')
            if len(tokens) < 1:
                continue
            tokens = tokenizer.convert_tokens_to_ids(tokens)
            nbatch = len(tokens) // seq_len
            for i in range(nbatch + 1):
                seqs.append(tokens[i * seq_len:(i + 1) * seq_len])
            if len(tokens) > nbatch * seq_len:
                seqs.append(tokens[nbatch * seq_len:])

    for i, seq in enumerate(seqs):
        nfeedback = len(seq)
        if nfeedback < 3:
            user_train[i] = seq
            user_valid[i] = []
            user_test[i] = []
        else:
            user_train[i] = seq[:-2]
            user_valid[i] = []
            user_valid[i].append(seq[-2])
            user_test[i] = []
            user_test[i].append(seq[-1])
    seq_num = len(seqs)
    item_num = len(tokenizer.vocab) - 5
    return [user_train, user_valid, user_test, seq_num, item_num]


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end=' ')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end=' ')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
