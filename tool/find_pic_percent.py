#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'my_pic_note'
__author__ = 'deagle'
__date__ = '11/25/2020 10:51'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
"""

from shutil import copyfile
from shutil import copy

from tool.Files import files

'''
把云事业部的标注文件和图片对应起来
'''
import datetime
import os

percent = 0.7


def main():
    json_folder = r'D:\work_source\CV_Project\datasets\dataresult_xi_an_20201125'
    pic_folder = r'D:\work_source\CV_Project\datasets\待标定_20201111'

    new_pic_folder = 'D:/work_source/CV_Project/datasets/xi_an_20201125/all/pictures'

    train_truth = 'D:/work_source/CV_Project/datasets/xi_an_20201125/train/truth'
    val_truth = 'D:/work_source/CV_Project/datasets/xi_an_20201125/val/truth'
    train_pic = 'D:/work_source/CV_Project/datasets/xi_an_20201125/train/pic'
    val_pic = 'D:/work_source/CV_Project/datasets/xi_an_20201125/val/pic'

    fs = files()

    json_list = fs.travel_folder(json_folder, '*')
    print("json file num: %s" % len(json_list))
    json_quantity_70p = int(len(json_list) * 0.7)
    print("json_quantity_70p: %s" % json_quantity_70p)

    current_file_num = 0;

    # 给标注文件做划分
    json_name_list = []
    train_list = []
    val_list = []
    for file in json_list:
        current_file_num = current_file_num + 1
        json_name_list.append(os.path.basename(file).replace('.json', ''))
        if current_file_num <= json_quantity_70p:
            train_list.append(os.path.basename(file).replace('.json', ''))
            sss = train_truth+'/'+os.path.basename(file)
            copy(file, train_truth+'/'+os.path.basename(file))
        else:
            val_list.append(os.path.basename(file).replace('.json', ''))
            copy(file, val_truth+'/'+os.path.basename(file))

    # 给输出图片路径生成文件夹
    if not os.path.exists(new_pic_folder):
        os.makedirs(new_pic_folder)
    if not os.path.exists(train_pic):
        os.makedirs(train_pic)
    if not os.path.exists(val_pic):
        os.makedirs(val_pic)

    # 遍历所有图片文件，路径存储到List
    pic_list = fs.travel_folder(pic_folder, 'jpg')
    print("pic file num: %s" % len(pic_list))

    for pic in pic_list:
        # pic_name = pic.split('\\')[len(pic.split('\\')) - 1]
        pic_name = os.path.basename(pic).replace('.jpg', '')
        if pic_name in json_name_list:
            new_file = new_pic_folder + '/' + pic_name + '.jpg'
            print(new_file)
            copyfile(pic, new_file)
        # train
        if pic_name in train_list:
            new_file = train_pic + '/' + pic_name + '.jpg'
            # print(new_file)
            copyfile(pic, new_file)
        # val
        if pic_name in val_list:
            new_file = val_pic + '/' + pic_name + '.jpg'
            # print(new_file)
            copyfile(pic, new_file)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('time cost: %s' % str(datetime.datetime.now() - start_time).split('.')[0])
