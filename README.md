# BERT4RS

### 1. 使用Movies_and_TV进行预训练：

| Dataset | #users | #items | avg_action/user | avg_action/item | #action|
| ------ | ------ | ------ |------ | ------ | ------ |
| Movies_and_TV | 123960 | 50052 | 8.7| 21.7 | 1084572 |

将数据集按1:1划分为pretrain和test，每部分包括0.5M action，pretrain用于BERT预训练，test用于fine-tune

使用[bert](https://github.com/google-research/bert)预训练

预训练结果：

```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  global_step = 500000
INFO:tensorflow:  loss = 1.1401069
INFO:tensorflow:  masked_lm_accuracy = 0.7652841
INFO:tensorflow:  masked_lm_loss = 1.1527702
INFO:tensorflow:  next_sentence_accuracy = 1.0
INFO:tensorflow:  next_sentence_loss = 3.410484e-05
```

### 2. Fine-tune

预训练模型[下载](https://pan.baidu.com/s/1eZH7ImirmMZkFZYEuWG0JA)，放在`data/Movies_and_TV_5`

SeqRec为代码目录，执行`python bert_main.py`

执行`tensorboard logdir=./log`查看结果

![](https://github.com/xmzzyo/BERT4RS/blob/master/xmz_2019-03-18_09-26-26.png)
