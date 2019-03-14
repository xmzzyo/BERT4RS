# BERT4RS

### 1. 使用Movies_and_TV进行预训练：

| Dataset | #users | #items | avg_action/user | avg_action/item | #action|
| ------ | ------ | ------ |------ | ------ | ------ |
| Movies_and_TV | 123960 | 50052 | 8.7| 21.7 | 1084572 |

将数据集按1:1划分为pretrain和test，每部分包括0.5M action，pretrain用于BERT预训练，test用于fine-tune

使用[bert](https://github.com/google-research/bert)预训练

预训练结果：

```
global_step = 20000
loss = 7.646792
masked_lm_accuracy = 0.10956605
masked_lm_loss = 7.6468687
next_sentence_accuracy = 1.0
next_sentence_loss = 0.0007768932
```

### 2. Fine-tune

预训练模型[下载](https://pan.baidu.com/s/1eZH7ImirmMZkFZYEuWG0JA)，放在`data/Movies_and_TV_5`

SeqRec为代码目录，执行`python bert_main.py`

执行`tensorboard logdir=./log`查看结果