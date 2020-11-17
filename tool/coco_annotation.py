# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import json
from collections import defaultdict
from tqdm import tqdm
import os

"""hyper parameters"""
json_file_path = r'D:\work_source\CV_Project\datasets\footbridge_20201111\val\coco.json'
images_dir_path = 'D:/work_source/CV_Project/datasets/footbridge_20201111/val/pic/'
output_path = '../label/footbridge_val.txt'

dict_output_path = '../label/dict_footbridge_val.txt'

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

# 生成一个字典文件给模型训练使用，因为在前馈环节需要一个tensor独有量，这个字典给后面的get img id 使用
img_dict = {}

"""generate labels"""
images = data['images']
annotations = data['annotations']
for ant in tqdm(annotations):
    id = ant['image_id']
    name = os.path.join(images_dir_path, images[id - 1]['file_name'])

    # 向字典中添加数据
    img_dict[name] = '{:012d}'.format(id)
    # name = os.path.join(images_dir_path, '{:012d}'.format(id)+'-'+images[id-1]['file_name'])
    # name = os.path.join(images_dir_path, '{:012d}.jpg'.format(id))
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])

"""把图片字典输出到文件"""
with open(dict_output_path, 'w') as dict_f:
    for d in img_dict:
        dict_f.write(d + '-' + img_dict[d] + '\n')

"""write to txt"""
with open(output_path, 'w') as f:
    for key in tqdm(name_box_id.keys()):
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
