#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'pytorch-YOLOv4'
__author__ = 'deagle'
__date__ = '11/15/2020 23:27'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import cv2

img_path = 'D:/work_source/CV_Project/datasets/footbridge_20201111/train2/pic/IMG_1824_4260.jpg'

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img)

def main():
    print()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
