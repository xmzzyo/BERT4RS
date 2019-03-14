# -*- coding: utf-8 -*-
import numpy as np
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch import nn

from dataset import random_neq


class BertForSeqRec(BertPreTrainedModel):
    """BERT model for sequential recommendation."""

    def __init__(self, config, voc_size):
        super(BertForSeqRec, self).__init__(config)
        self.config = config
        self.voc_size = voc_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.vocab_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, pos_ids=None, neg_ids=None, input_mask=None, maxlen=10):
        sequence_output, _ = self.bert(input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        # sequence_output = sequence_output[1:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if neg_ids is not None:
            loss = self.cross_entropy(sequence_output, pos_ids, neg_ids, input_mask, maxlen)
            return loss
        else:
            test_logits = self.predict(sequence_output, pos_ids, input_mask, maxlen)
            return self.get_metric(test_logits)

    def cross_entropy(self, seq_out, pos_ids, neg_ids, mask, maxlen):

        pos_emb = self.bert.embeddings(pos_ids)
        neg_emb = self.bert.embeddings(neg_ids)
        # print("pos/neg emb shape: ", pos_emb.shape)

        pos = pos_emb.view(pos_emb.size(0) * maxlen, -1)
        neg = neg_emb.view(neg_emb.size(0) * maxlen, -1)
        # print("pos/neg shape: ", pos.shape)

        seq_emb = seq_out.view(-1, self.config.hidden_size)
        # print("seq_emb shape: ", seq_emb.shape)

        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)
        # print("pos/neg logits shape: ", pos_logits.shape)

        istarget = mask.view(mask.size(0) * maxlen).float()
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        return loss

    def predict(self, seq_out, pos_ids, mask, maxlen):

        test_item = []
        for pos, mask_ in zip(pos_ids, mask):
            pos = pos.masked_select(mask_)
            rated = set(pos)
            test = [pos[-1].item()]
            test.extend([random_neq(5, self.config.vocab_size, rated) for _ in range(self.voc_size - 1)])
            test_item.append(test)
        test_item = torch.LongTensor(test_item).cuda()
        # seq_emb = seq_out.view(-1, self.config.hidden_size)
        test_item_emb = self.bert.embeddings(test_item)
        test_logits = torch.matmul(seq_out, test_item_emb.transpose(1, 2))
        # print("test_logits shape: ", test_logits.shape)
        test_logits = test_logits.view(pos_ids.size(0), maxlen, self.voc_size)
        # print("test_logits shape: ", test_logits.shape)
        test_logits = test_logits[:, -1, :]
        # print("test_logits shape: ", test_logits.shape)
        return test_logits

    def get_metric(self, test_logists):
        NDCG = 0.0
        HIT = 0.0
        # print(test_logists.shape)
        # print(test_logists.argsort().argsort().shape)
        ranks = test_logists.argsort(descending=True).argsort()[:, 0].cpu()
        # print(ranks)
        # print(ranks)
        for rank in ranks:
            if rank < 10:
                NDCG += 1.0 / np.log2(rank + 2.0)
                HIT += 1.0
        return {"NDCG": NDCG / ranks.size(0), "HIT": HIT / ranks.size(0)}
