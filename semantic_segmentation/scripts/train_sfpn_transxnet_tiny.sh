#!/usr/bin/env bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=$((RANDOM+10000)) \
train.py \
configs/sfpn_transxnet_tiny.py \
--work-dir work_dirs/sfpn_transxnet_tiny/ \
--launcher pytorch