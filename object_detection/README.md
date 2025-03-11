# Applying TransXNet to Object Detection and Instance Segmentation

For details, please address "[TransXNet: Learning Both Global and Local Dynamics with a Dual Dynamic Token Mixer for Visual Recognition](https://arxiv.org/abs/2310.19380)".   

## 1. Requirements
```
# Environments:
cuda==11.3
python==3.8.15
# Packages:
mmcv==1.7.1
mmdet==2.28.2
timm==0.6.12
torch==1.12.1
torchvision==0.13.1
```


## 2. Data Preparation

Prepare COCO 2017 according to the [guidelines](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/1_exist_data_model.md).  

## 3. Main Results on COCO with Pretrained Models


| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|------------------------------------------------------|----------|
| RetinaNet  | TransXNet-T | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-t.pth.tar) |    1x   |  No |  43.1  |    -    | [config](configs/retinanet_transx_t_fpn_1x_coco.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/retinanet_tiny.log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/retinanet_tiny.pth) |
| RetinaNet  | TransXNet-S | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-s.pth.tar) |    1x   |  No |  46.4  |    -    | [config](configs/retinanet_transx_s_fpn_1x_coco.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/retinanet_small.log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/retinanet_small.pth) |
| RetinaNet  | TransXNet-B | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-b.pth.tar) |    1x   |  No |  47.6  |    -    | [config](configs/retinanet_transx_b_fpn_1x_coco.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/reinanet_base.log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/retinanet_base.pth) |
| Mask R-CNN | TransXNet-T | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-t.pth.tar) |    1x   |  No |  44.5 |   40.7  | [config](configs/mask_rcnn_transx_t_fpn_1x_coco.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/maskrcnn_tiny.log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/maskrcnn_tiny.pth) |
| Mask R-CNN | TransXNet-S | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-s.pth.tar) |    1x   |  No |  47.7  |   43.1  | [config](configs/mask_rcnn_transx_s_fpn_1x_coco.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/maskrcnn_small.log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/maskrcnn_small.pth) |
| Mask R-CNN | TransXNet-B | [ImageNet-1K](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/transx-b.pth.tar) |    1x   |  No |  48.8  |   43.8  | [config](configs/mask_rcnn_transx_b_fpn_1x_coco.py) | [log](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/maskrcnn_base.log.json) & [model](https://github.com/LMMMEng/TransXNet/releases/download/v1.0/maskrcnn_base.pth) |


## 4. Train
To train ``TransXNet-T + RetinaNet`` models on COCO train2017 with 8 gpus (single node), run:
```
bash dist_train.sh configs/retinanet_transx_t_fpn_1x_coco.py 8
```
To train ``TransXNet-T + Mask R-CNN`` models on COCO train2017 with 8 gpus (single node), run:
```
bash dist_train.sh configs/mask_rcnn_transx_t_fpn_1x_coco.py 8
```

## 5. Validation
To evaluate ``TransXNet-T + RetinaNet`` models on COCO val2017, run:
```
bash dist_test.sh configs/retinanet_transx_t_fpn_1x_coco.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```
To evaluate ``TransXNet-T + Mask R-CNN`` models on COCO val2017, run:
```
bash dist_test.sh configs/mask_rcnn_transx_t_fpn_1x_coco.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox segm
```

## Citation
If you find this project useful for your research, please consider citing:
```
@article{lou2023transxnet,
  title={TransXNet: Learning Both Global and Local Dynamics with a Dual Dynamic Token Mixer for Visual Recognition},
  author={Meng Lou and Shu Zhang and Hong-Yu Zhou and Chuan Wu and Sibei Yang and Yizhou Yu},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025}
}
```

## Contact
If you have any questions, please feel free to [create issues](https://github.com/LMMMEng/TransXNet/issues) or contact me at lmzmm.0921@gmail.com.
