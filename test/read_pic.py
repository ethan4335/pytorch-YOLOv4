#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'pytorch-YOLOv4'
__author__ = 'deagle'
__date__ = '12/4/2020 10:31'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import cv2

def main():
    img = cv2.imread('D:/work_source/CV_Project/datasets/test_pictures/test2.jpg')
    sized = cv2.resize(img, (608, 608))
    print()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print(str(datetime.datetime.now() - start_time).split('.')[0])
