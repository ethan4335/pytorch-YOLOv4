#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'pytorch-YOLOv4'
__author__ = 'deagle'
__date__ = '11/23/2020 11:30'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime
from tensorboardX import SummaryWriter


def main():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)
    writer.close()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
