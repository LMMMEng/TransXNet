#!/usr/bin/env bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=$((RANDOM+6666)) \
test.py \
configs/sfpn_transxnet_small.py \
/mnt/users/Practice/ConvFormer/poolformer-main/sem_seg/work_dirs/sfpn/sfpn_convformer_small_dpr0.15_RTX/iter_80000.pth \
--out work_dirs/sfpn_transxnet_small/latest.pkl \
--eval mIoU \
--launcher pytorch \
# --aug-test