# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import Dataset

logging.basicConfig(filename='log.txt', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", data_type="train"):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.data_type = data_type
        # load samples into memory
        self.all_docs = []
        self.doc = []
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm(f.readlines(), desc="Loading Dataset"):
                line = line.strip()
                tokens = self.tokenizer.tokenize(line)
                if len(tokens) < 5:
                    continue
                nbatch = len(tokens) // self.seq_len
                for i in range(nbatch):
                    self.doc.append(tokens[len(tokens) - (i + 1) * self.seq_len:len(tokens) - i * self.seq_len])
                # if len(tokens) > nbatch * self.seq_len + 5:
                #     self.doc.append(tokens[0:len(tokens) - nbatch * self.seq_len])
                # self.all_docs.append(self.doc)
                # self.doc = []

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, index):

        tokens = self.doc[index]

        assert self.data_type in {"train", "valid", "test"}
        if self.data_type == "train":
            tokens = tokens[:-2]
        elif self.data_type == "valid":
            tokens = tokens[:-1]

        # combine to one sample
        cur_example = InputExample(guid=index, tokens=tokens)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.data_type)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.pos_ids),
                       torch.tensor(cur_features.neg_ids),
                       torch.tensor(cur_features.input_mask, dtype=torch.uint8))

        return cur_tensors


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens):
        """Constructs a InputExample."""
        self.guid = guid
        self.tokens = tokens


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, pos_ids, neg_ids):
        self.input_ids = input_ids
        self.pos_ids = pos_ids
        self.neg_ids = neg_ids
        self.input_mask = input_mask


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def convert_example_to_features(example, max_seq_length, tokenizer, data_type):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :param data_type:
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"]
    for token in example.tokens:
        tokens.append(token)

    # TODO whether need to add SEP
    # tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    ts = set(input_ids)

    pos_ids = [0] + input_ids[2:]
    input_ids = input_ids[:-1]
    # start from 5 to avoid cls sep pad unk mask
    neg_ids = [random_neq(5, len(tokenizer.vocab), ts) for _ in range(len(pos_ids))]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    input_mask[0] = 0

    # Zero-pad up to the sequence length.
    input_pad = [0] * (max_seq_length - len(input_ids))
    input_ids = input_pad + input_ids

    mask_pad = [0] * (max_seq_length - len(input_mask))
    input_mask = mask_pad + input_mask

    pos_pad = [0] * (max_seq_length - len(pos_ids))
    pos_ids = pos_pad + pos_ids

    neg_pad = [0] * (max_seq_length - len(neg_ids))
    neg_ids = neg_pad + neg_ids

    assert len(input_ids) == max_seq_length
    assert len(pos_ids) == max_seq_length
    assert len(neg_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("pos_ids: %s" % " ".join([str(x) for x in pos_ids]))
        logger.info("neg_ids: %s" % " ".join([str(x) for x in neg_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))

    features = InputFeatures(input_ids=input_ids,
                             pos_ids=pos_ids,
                             neg_ids=neg_ids,
                             input_mask=input_mask)
    return features
