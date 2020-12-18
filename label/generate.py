#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'pytorch-YOLOv4'
__author__ = 'deagle'
__date__ = '12/16/2020 20:34'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
读取多图片路径和label路径
生成训练和验证数据
2:8比例拆分

eg.
D:/work_source/CV_Project/datasets/footbridge_20201111/val_two/pic/IMG_2569.JPG 1459,1176,1953,2179,0 1681,1975,1817,2029,1
"""

import datetime
import os
import json


class travel_folder():
    def __init__(self):
        pass

    def get_file_list(self, folder):
        f_list = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                f = os.path.join(root, file)
                f_list.append(f)
        return f_list

    def get_json_list(self, folder):
        j_list = []
        for root, files in os.walk(folder):
            for file in files:
                if '.json' in file:
                    j = os.path.join(root, file)
                    j_list.append(j)
        return j_list

    def get_json_short_name_list(self, folder):
        json_short_name_list = []
        for root, files in os.walk(folder):
            for file in files:
                if '.json' in file:
                    j = os.path.join(root, str(file).replace('.json', ''))
                    json_short_name_list.append(j)
        return json_short_name_list

    def get_img_list(self, folder):
        img_list = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if '.png' in file or '.jpg' in file:
                    img = os.path.join(root, file)
                    img_list.append(img)
        return img_list

    def get_img_dict(self, folder):
        img_dict = {}
        for root, dirs, files in os.walk(folder):
            for file in files:
                if '.png' in file or '.jpg' in file:
                    img = os.path.join(root, file)
                    name = file.split('.')[0]
                    img_dict[img] = name
        return img_dict


class decode_json():
    def __init__(self):
        pass

    def json_to_dict(self,j):
        with open(j, encoding='utf-8') as f:
            img_name = os.path.basename(j).replace('.json', '') + '.jpg'
            data = json.load(f)
            annotation_new = []
            objects = data['objects']
            have_target = False
            annotation_new.append(pic_folder + '/' + str(img_name))
            for obj in objects:
                obj_points = obj['obj_points']
                f_code = obj['f_code']
                f_code_index = 0
                if f_code in f_code_dict:
                    f_code_index = f_code_dict[f_code]
                else:
                    len1 = len(f_code_dict)
                    f_code_dict[f_code] = len1
                f_name = obj['f_name']
                name_set_cn[str(f_code)] = f_name
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
                        obj_new = str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(f_code_dict[f_code])
                        annotation_new.append(obj_new)

            # write one pic annotation to file
            ana_str = ''
            if have_target:
                for ana in annotation_new:
                    ana_str = ana_str + str(ana) + ' '
                annotations.append(ana_str)

            if have_target is not True:
                try:
                    os.remove(os.path.join(pic_folder, img_name))
                except:
                    pass
            # dict
            img_dict[img_name] = '{:012d}'.format(file_num - 1)



def main():
    img_folder = r''
    json_folder = r''


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('-' * 20)
    print('time cost: %s' % str(datetime.datetime.now() - start_time).split('.')[0])
