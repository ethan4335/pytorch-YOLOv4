# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg_footbridge = EasyDict()

Cfg_footbridge.use_darknet_cfg = True
Cfg_footbridge.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4_footbridge.cfg')

Cfg_footbridge.batch = 3
Cfg_footbridge.subdivisions = 1
Cfg_footbridge.width = 608
Cfg_footbridge.height = 608
Cfg_footbridge.channels = 3
Cfg_footbridge.momentum = 0.949
Cfg_footbridge.decay = 0.0005
Cfg_footbridge.angle = 0
Cfg_footbridge.saturation = 1.5
Cfg_footbridge.exposure = 1.5
Cfg_footbridge.hue = .1

Cfg_footbridge.learning_rate = 0.00261
Cfg_footbridge.burn_in = 1000
Cfg_footbridge.max_batches = 500500
Cfg_footbridge.steps = [400000, 450000]
Cfg_footbridge.policy = Cfg_footbridge.steps
Cfg_footbridge.scales = .1, .1

Cfg_footbridge.cutmix = 0
Cfg_footbridge.mosaic = 1

Cfg_footbridge.letter_box = 0
Cfg_footbridge.jitter = 0.2
Cfg_footbridge.classes = 4
Cfg_footbridge.track = 0
Cfg_footbridge.w = Cfg_footbridge.width
Cfg_footbridge.h = Cfg_footbridge.height
Cfg_footbridge.flip = 1
Cfg_footbridge.blur = 0
Cfg_footbridge.gaussian = 0
Cfg_footbridge.boxes = 60  # box num 60
Cfg_footbridge.TRAIN_EPOCHS = 1
Cfg_footbridge.train_label = os.path.join(_BASE_DIR, 'label', 'footbridge_train_mini.txt')
Cfg_footbridge.val_label = os.path.join(_BASE_DIR, 'label' ,'footbridge_val.txt')
Cfg_footbridge.TRAIN_OPTIMIZER = 'adam'
# ethan add
Cfg_footbridge.dataset_dir='D:/work_source/CV_Project/datasets/footbridge_20201111/train_mini/pic'
Cfg_footbridge.pretrained = './pretrained/yolov4.conv.137.pth'
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg_footbridge.mosaic and Cfg_footbridge.cutmix:
    Cfg_footbridge.mixup = 4
elif Cfg_footbridge.cutmix:
    Cfg_footbridge.mixup = 2
elif Cfg_footbridge.mosaic:
    Cfg_footbridge.mixup = 3

Cfg_footbridge.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg_footbridge.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg_footbridge.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg_footbridge.keep_checkpoint_max = 20
