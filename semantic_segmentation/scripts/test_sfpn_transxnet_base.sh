#!/usr/bin/env bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=$((RANDOM+8888)) \
test.py \
configs/sfpn_transxnet_base.py \
/mnt/users/Practice/ConvFormer/poolformer-main/sem_seg/work_dirs/sfpn/sfpn_convformer_base/best_mIoU_iter_72000.pth \
--out work_dirs/sfpn_transxnet_base/latest.pkl \
--eval mIoU \
--launcher pytorch \
# --aug-test