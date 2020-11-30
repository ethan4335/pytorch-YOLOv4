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
import os

"""hyper parameters"""
ni_json_folder = r'D:\work_source\CV_Project\datasets\dataresult_xi_an_20201125'
pic_folder = 'D:/work_source/CV_Project/datasets/xi_an_20201125'
new_jspn_file = r'D:\work_source\CV_Project\datasets\xi_an_20201125\annotations_xi_an_20201125.txt'
new_name_file = r'D:\work_source\CV_Project\datasets\xi_an_20201125\names_xi_an_20201125.txt'
new_name_cn_file = r'D:\work_source\CV_Project\datasets\xi_an_20201125\names_xi_an_20201125_cn.txt'

# read pic file
# for root, dirs, files in os.walk(pic_folder):
#     for pic in files:  # file name not dir
#         pass

# read navinfo json file
annotations = []
# 可以使用大括号 { } 或者 set() 函数创建集合，注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典
name_set = set()
name_set2 = {}

for root, dirs, files in os.walk(ni_json_folder):
    for file in files:
        file_json = os.path.join(root, file)
        with open(file_json, encoding='utf-8') as f:
            img_name = os.path.basename(file_json).replace('.json', '') + '.jpg'
            data = json.load(f)
            annotation_new = []
            objects = data['objects']
            have_target = False
            annotation_new.append(str(img_name))
            for obj in objects:
                obj_points = obj['obj_points']
                f_code = obj['f_code']
                f_name = obj['f_name']
                name_set2[str(f_code)] = f_name
                name_set.add(f_code)
                for point in obj_points:
                    if 'w' in point:
                        have_target = True
                        w = point['w']
                        x = point['x']
                        h = point['h']
                        y = point['y']
                        x1 = int(x)
                        y1 = int(y)
                        x2 = x1 + int(w)
                        y2 = y1 + int(h)
                        obj_new = str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(f_code)
                        annotation_new.append(obj_new)

            # write one pic annotation to file
            ana_str = ''
            if have_target:
                for ana in annotation_new:
                    ana_str = ana_str + str(ana) + ' '
                annotations.append(ana_str)

with open(new_jspn_file, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    for line in annotations:
        f.write(str(line.strip()).replace('\'', '') + '\n')
        # f.newlines()

with open(new_name_file, 'w') as f:
    for line in name_set:
        f.write(str(line) + '\n')

with open(new_name_cn_file, 'w') as f:
    for key in name_set2.keys():
        f.write(format(key) + ':' + name_set2[key] + '\n')
