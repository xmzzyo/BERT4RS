# -*- coding: utf-8 -*-
import numpy as np
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch import nn

from dataset import random_neq


class BertForSeqRec(BertPreTrainedModel):
    """BERT model for sequential recommendation.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, voc_size):
        super(BertForSeqRec, self).__init__(config)
        self.config = config
        self.voc_size = voc_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, voc_size)
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
