#!/usr/bin/env bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
train.py \
/path/to/imagenet/ \
--batch-size 128 \
--pin-mem \
--model transxnet_b \
--drop-path 0.4 \
--lr 2e-3 \
--warmup-epochs 5 \
--sync-bn \
--model-ema \
--model-ema-decay 0.99985 \
--val-start-epoch 250 \
--val-freq 50 \
--native-amp \
--output /path/to/save-checkpoint/