# -*- coding: utf-8 -*-
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.tokenization import whitespace_tokenize
from torch import Tensor
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm
import tensorflow as tf

from bert_model import BertForSeqRec
from dataset import BERTDataset


class RSTokenizer(BertTokenizer):
    def tokenize(self, text):
        return whitespace_tokenize(text)


def test_dataset():
    tokenizer = RSTokenizer.from_pretrained("../../data/Movies_and_TV_5/ckp", do_lower_case=False)

    train_dataset = BERTDataset("../../data/data_Movies_and_TV_5_10_core.txt", tokenizer, seq_len=50)
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, pos_ids, neg_ids, input_mask = batch


def test_state_dict():
    model = BertForSeqRec.from_pretrained("../../data/Movies_and_TV_5/ckp", voc_size=101)
    weights_path = "../../data/Movies_and_TV_5/ckp/pytorch_model.bin"
    state_dict = torch.load(weights_path)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')
    # print(missing_keys)
    for name, param in model.named_parameters():
        key = name
        if key in state_dict:
            pass
        else:
            print(key)
    # print(state_dict.keys())
    # for key, value in model.named_parameters():
    #     print(key)


def test_multi():
    x = torch.randn(3, 2)
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.randn(3, 2)
    y = torch.tensor([[1, 3], [2, 4]])

    print(x)
    print(y)
    # print(torch.matmul(x, y))
    print(x * y)


def test_mask_select():
    x = torch.randn(4, 3, 2)
    mask = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1]])
    # mask.unsqueeze(2)
    # print(mask)
    print(x)
    print(x.transpose(0, 2))
    # print(x.masked_select(mask))


def test_argsort():
    x = np.array([[0.5, 0.3, 0.2, 0.3, 0.1, 0.1],
                  [0.1, 0.3, 0.2, 0.3, 0.1, 0.1]])
    x = torch.tensor(x)
    print(x)
    print(x.argsort(descending=True))
    print(x.argsort(descending=True).argsort())


def test_token():
    seq_len = 10
    doc = []
    tokens = [13293, 8253, 13391, 26096, 24713, 25456, 49687, 31900, 3401, 18506, 24062, 18989, 6653, 10015, 34766,
              2579, 19490, 37311, 45389, 39034, 342, 19268, 26823, 10472, 34093, 30847, 39763, 47911, 38727, 26780,
              27515, 9687, 36553, 40631, 27582, 43148, 49344, 15350, 41604, 7252, 6420, 1183, 39709, 5376, 21678, 34844,
              35289, 20120, 33456, 14551, 12259, 12576, 41593, 33197, 38790, 34499, 46856, 30429, 2310, 35149, 4437,
              47320, 30066, 17582, 49507, 33493, 11074, 13600, 20753, 25772, 22412, 13710, 24557, 9405, 46877, 22097,
              14677, 21057, 20666, 28802, 14434, 13708, 44565, 10494, 1859, 21642, 7013, 17218, 15108, 6482, 22157, 190,
              48091, 7412, 29406, 29592, 28719, 36119, 9886, 15469, 10243]
    nbatch = len(tokens) // seq_len
    for i in range(nbatch):
        doc.append(tokens[len(tokens) - (i + 1) * seq_len:len(tokens) - i * seq_len])
    print(doc)


def convert_SASRec():
    tokenizer = RSTokenizer.from_pretrained("../../data/Movies_and_TV_5/ckp_v1/", do_lower_case=False)
    fname = "../../data/Movies_and_TV_5/test_Movies_and_TV_5.txt"
    seqs = []
    with open(fname, "r") as f:
        for line in f.readlines():
            seqs.append(line.strip().split())
    u = 1
    with open("../../data/Movies_and_TV_5/SASRec_Movies_and_TV_5.txt", "w") as f:
        for seq in seqs:
            input_ids = tokenizer.convert_tokens_to_ids(seq)
            for a in input_ids:
                f.write("%s %s\n" % (str(u), str(a - 4)))
            u += 1


def test_crossentropy():
    loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    input = torch.FloatTensor([[0, 1, 2, 3, 4, 0], [2, 3, 4, 5, 0, 0]])
    target = torch.LongTensor([0, 0])
    output = loss(input, target)
    print(output.item())


if __name__ == "__main__":
    test_crossentropy()
