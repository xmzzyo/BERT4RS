# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse

from pytorch_pretrained_bert import BertConfig
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer, whitespace_tokenize
from pytorch_pretrained_bert.optimization import BertAdam

import random

from bert_model import BertForSeqRec
from dataset import BERTDataset

logging.basicConfig(filename='log.txt', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class RSTokenizer(BertTokenizer):
    def tokenize(self, text):
        return whitespace_tokenize(text)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default='../data/Movies_and_TV_5/test_Movies_and_TV_5.txt',
                        type=str,
                        help="The input train corpus.")
    parser.add_argument("--bert_model",
                        default='../data/Movies_and_TV_5/ckp_v1/',
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='../data/Movies_and_TV_5/ckp_v1_1',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=50,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1000.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        default=True,
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    summary_writer = SummaryWriter("log")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = RSTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # train_examples = None
    num_train_steps = None
    if args.do_train:
        print("Loading Train/Valid/Test Dataset", args.train_file)
        train_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length)
        valid_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length, data_type="valid")
        test_dataset = BERTDataset(args.train_file, tokenizer, seq_len=args.max_seq_length, data_type="test")

        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    print("Train/Valid/Test DataSet loaded...")
    # Prepare model
    voc_size = 101
    model = BertForSeqRec.from_pretrained(args.bert_model, voc_size=voc_size)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = RandomSampler(valid_dataset)
            test_sampler = RandomSampler(valid_dataset)
        else:
            # TODO: check if this works with current data generator from disk that relies on file.__next__
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        # train_dataset = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        valid_dataloadr = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

        for eph in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Train Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, pos_ids, neg_ids, input_mask = batch
                loss = model(input_ids, pos_ids, neg_ids, input_mask, args.max_seq_length)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                if step % 50 == 0:
                    logger.info('loss：' + str(tr_loss / (step + 1)))
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            summary_writer.add_scalar("train/loss", tr_loss / (nb_tr_steps + 1), eph + 1)

            model.eval()
            NDCG = 0.0
            HIT = 0.0
            eval_step = 0
            for step, batch in enumerate(tqdm(valid_dataloadr, desc="Eval Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, pos_ids, neg_ids, input_mask = batch
                metrics = model(input_ids, pos_ids, None, input_mask, args.max_seq_length)
                NDCG += metrics["NDCG"].item()
                HIT += metrics["HIT"]
                eval_step += 1
            logger.info("Epoch : %d" % eph)
            logger.info("Valid NDCG: %f" % (NDCG / eval_step))
            logger.info("Valid HIT: %f" % (HIT / eval_step))
            summary_writer.add_scalar("valid/NDCG@10", (NDCG / eval_step), eph + 1)
            summary_writer.add_scalar("valid/HIT@10", (HIT / eval_step), eph + 1)

            NDCG = 0.0
            HIT = 0.0
            test_step = 0
            for step, batch in enumerate(tqdm(test_dataloader, desc="Test Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, pos_ids, neg_ids, input_mask = batch
                metrics = model(input_ids, pos_ids, None, input_mask, args.max_seq_length)
                NDCG += metrics["NDCG"].item()
                HIT += metrics["HIT"]
                test_step += 1
            logger.info("Epoch : %d" % eph)
            logger.info("Test NDCG: %f" % (NDCG / test_step))
            logger.info("Test HIT: %f" % (HIT / test_step))
            summary_writer.add_scalar("test/NDCG@10", (NDCG / eval_step), eph + 1)
            summary_writer.add_scalar("test/HIT@10", (HIT / eval_step), eph + 1)

            # Save a trained model
            if eph % 100 == 0:
                logger.info("** ** * Saving fine - tuned model ** ** * ")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model" + str(eph) + ".bin")
                if args.do_train:
                    torch.save(model_to_save.state_dict(), output_model_file)
        summary_writer.export_scalars_to_json("scalars.json")
        summary_writer.close()


if __name__ == "__main__":
    main()
