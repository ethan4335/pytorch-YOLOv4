#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'pytorch-YOLOv4'
__author__ = 'deagle'
__date__ = '11/16/2020 0:09'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""
import datetime



import os

lv, no = os.path.splitext(os.path.basename('D:sss/level1_123.jpg'))[0].split("_")
lv = lv.replace("level", "")
no = f"{int(no):04d}"
print(int(lv + no))


def main():
    print()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(str(time_cost).split('.')[0])
