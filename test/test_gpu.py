#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'pytorch-YOLOv4'
__author__ = 'deagle'
__date__ = '11/23/2020 15:17'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
import torch


def main():
    print('use: ',torch.cuda.is_available())


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
