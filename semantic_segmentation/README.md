# Applying TransXNet to Semantic Segmentation

For details, please address "[TransXNet: Learning Both Global and Local Dynamics with a Dual Dynamic Token Mixer for Visual Recognition](https://arxiv.org/abs/2310.19380)". 

## 1. Requirements

We highly suggest using our provided dependencies to ensure reproducibility:   
```
# Environments:
cuda==11.3
python==3.8.15
# Packages:
mmcv==1.7.1
timm==0.6.12
torch==1.12.1
torchvision==0.13.1
mmsegmentation==0.30.0
```

## 2. Data Preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets).


## 3. Main Results on ADE20K with Pretrained Models

| Method | Backbone | Pretrain | Iters | mIoU | Config | Download |
| --- | --- | --- |:---:|:---:| --- | --- |
| Semantic FPN | TransXNet-T   | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-t.pth.tar) |  80K  |     45.5    | [config](configs/sfpn_transxnet_tiny.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/sfpn_transxnet_tiny_log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/sfpn_transxnet_tiny.pth) |
| Semantic FPN | TransXNet-S  | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-s.pth.tar) |  80K  |     48.5    | [config](configs/sfpn_transxnet_small.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/sfpn_transxnet_small_log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/sfpn_transxnet_small.pth) |
| Semantic FPN | TransXNet-B | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-b.pth.tar) |  80k  |     49.9    | [config](configs/sfpn_transxnet_base.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/sfpn_transxnet_base_log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/sfpn_transxnet_base.pth) |


## 4. Train
To train ``TransXNet + Semantic FPN`` models on ADE20K with 8 gpus (single node), run:
```
bash scripts/train_sfpn_transxnet_tiny.sh # train TransXNet-T + Semantic FPN
bash scripts/train_sfpn_transxnet_small.sh # train TransXNet-S + Semantic FPN
bash scripts/train_sfpn_transxnet_base.sh # train TransXNet-B + Semantic FPN
```

## 5. Validation
To evaluate ``TransXNet + Semantic FPN`` models on ADE20K, run:
```
# Take TransXNet-T + Semantic FPN as an example:
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=$((RANDOM+8888)) \
test.py \
configs/sfpn_transxnet_tiny.py \
path/to/checkpoint \
--out work_dirs/output.pkl \
--eval mIoU \
--launcher pytorch
```

## Citation
If you find this project useful for your research, please consider citing:
```
@article{lou2023transxnet,
  title={TransXNet: Learning Both Global and Local Dynamics with a Dual Dynamic Token Mixer for Visual Recognition},
  author={Lou, Meng and Zhou, Hong-Yu and Yang, Sibei and Yu, Yizhou},
  journal={arXiv preprint arXiv:2310.19380},
  year={2023}
}
```

## Contact
If you have any questions, please feel free to [create issues](https://github.com/LMMMEng/TransXNet/issues) or contact me at lmzmm.0921@gmail.com.
