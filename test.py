# -*- coding: utf-8 -*-

import json
import random

import numpy as np
import torch
from pytorch_pretrained_bert import BertConfig
from sklearn.metrics.pairwise import cosine_similarity

# state_dict = torch.load('data\\Digital_Music_5\\checkpoint\\pytorch_model.bin')
#
# print(state_dict.keys())

# exit()


with open('data/Movies_and_TV_5/output.jsonl', 'r') as f:
    hidden = dict()
    for line in f.readlines():
        features = json.loads(line)
        for token in features['features']:
            token_name = token['token']
            layers = token['layers']
            values = []
            for i in layers:
                values.extend(i['values'])
            hidden[token_name] = values

print(len(hidden['6301977467']))
print(cosine_similarity([hidden['B00VAV2JH0'], hidden['B01EH4SV7S']]))
print(np.array(hidden['6301977467']) - np.array(hidden['B00JAQJMJ0']))

exit()

config = BertConfig(
    vocab_size_or_config_json_file=3573,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=1024,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=128)
with open('bert_config.json', 'w') as f:
    f.write(config.to_json_string())

exit()

time_interva = [-200, -110, 0, 23, 25, 54, 67, 90, 102, 104, 108, 190, 200]
time_interval = [time_interva[i] - time_interva[i - 1] for i in range(1, len(time_interva))]
time_interval.insert(0, 0)
print(time_interval)
split_idx = [0, len(time_interval)]
has_max_length = True
split_idx.sort()
i = 0
max_length = 4
min_length = 2
while i < len(split_idx) - 1:
    split_idx.sort()
    start = split_idx[i]
    end = split_idx[i + 1]
    len_tmp = end - start
    if len_tmp > max_length:
        max_idx = np.argmax(time_interval[start:end])
        tim_tmp = time_interval[start: end]
        tim_tmp.sort(reverse=True)
        for j in range(len_tmp):
            if max_idx < min_length or len_tmp - max_idx < min_length:
                max_idx = time_interval.index(tim_tmp[j]) - start
                print(max_idx)
            else:
                break
        split_idx.append(max_idx + split_idx[i])
        time_interval[max_idx + split_idx[i]] = -1
        i -= 1
    i += 1
split_idx.sort()
print(split_idx)
print(time_interval)
seq = []
for i in range(len(split_idx) - 1):
    print(split_idx[i], split_idx[i + 1])
    seq.append(time_interva[split_idx[i]:split_idx[i + 1]])
print(seq)
