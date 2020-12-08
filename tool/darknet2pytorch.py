import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool.region_loss import RegionLoss
from tool.yolo_layer import YoloLayer
from tool.config import *
from tool.torch_utils import *

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        '''
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x


class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)
        
        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous().\
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x


class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        out = F.interpolate(x, size=(x.size(2) * self.stride, x.size(3) * self.stride), mode='nearest')
        return out


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
# 用 members、blocks、net_info 和 module_list 对网络进行初始化。
# pytorch 加载训练好的模型做inference
class Darknet(nn.Module):
    def __init__(self, cfgfile, inference=False):
        super(Darknet, self).__init__()
        self.inference = inference  # 预加载模型，类似于tf中的pretrain
        self.training = not self.inference  # 没有预加载的模型，那么需要重新train

        """一共有163个block，除去第一个block是网络的一些参数，其他162个block正好对应了162层yolo结构"""
        self.blocks = parse_cfg(cfgfile)    # 解析配置文件 获得模型结构，
        self.width = int(self.blocks[0]['width'])   # 608
        self.height = int(self.blocks[0]['height']) # 608

        # print('input size: %s'%self.width)

        '''block 有 163 个blocks, yolov4 一共161层'''
        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        # print(len(self.models) - 1)
        # print(self.models[0])
        # print(self.models[161])
        # !!! self.loss = YoloLayer
        self.loss = self.models[len(self.models) - 1]

        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0


# 一，计算输出；二，尽早处理的方式转换输出检测特征图（例如转换之后，这些不同尺度的检测图就能够串联，不然会因为不同维度不可能实现串联）。
    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []
        # 这里，我们迭代 self.block[1:] 而不是 self.blocks，因为 self.blocks 的第一个元素是一个 net 块，它不属于前向传播。
        # 源代码： modules = self.blocks[1:]
        for block in self.blocks:
            ind = ind + 1
            # if ind > 0:
            #    return x
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        x = outputs[layers[0]]
                        outputs[ind] = x
                    else:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
                        outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print("rounte number > 2 ,is {}".format(len(layers)))

            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'yolo':
                # if self.training:
                #     pass
                # else:
                #     boxes = self.models[ind](x)
                #     out_boxes.append(boxes)
                boxes = self.models[ind](x)
                out_boxes.append(boxes)
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))

        if self.training:
            return out_boxes
        else:
            return get_region_boxes(out_boxes)

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):

        """
        nn.ModuleList仅仅类似于pytho中的list类型，只是将一系列层装入列表，并没有实现forward()方法，因此也不会有网络模型产生的副作用，
        但它有和纯python中的list不同，用nn.ModuleList或者ParameterList包裹模型各个层之后，我们不光可以像python里的list一样对模型的各个层进行索引，
        同时这些层的参数将会被自动注册，这些层的参数只有被正确注册之后，优化器才能发现和训练这些参数
        这个操作只会生成一个空的list
        """
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:    # 遍历 blocks 的时候，模型不断累加到一起
            if block['type'] == 'net':  #网络参数
                prev_filters = int(block['channels']) #第一次的卷积核个数就是看看有多少个通道
                continue    # 一共就一个net参数表
            elif block['type'] == 'convolutional':  #卷积层
                conv_id = conv_id + 1
                # 数据经过多层神经网络后,数据分布发生变化,导致各层参数需要不到调整适应分布变化,这会让收敛速度变慢,
                # 超参数设定也变得比较复杂,这在论文里作者称作Internal Covariate Shift
                # batch_normalize 用于防止梯度消失，就是将后面的数据分布进行规整，
                batch_normalize = int(block['batch_normalize']) # always 1, 除了最后的输出层，每一层都是用了 batch_normalize
                filters = int(block['filters']) # 卷积核，其实就是滤波器。在每次迭代中会进行更新参数
                kernel_size = int(block['size'])
                stride = int(block['stride'])   # 步长
                is_pad = int(block['pad'])  # padding 过程 这里的padding 始终=1，仅用于表示：需要 padding
                pad = (kernel_size - 1) // 2 if is_pad else 0   # padding 大小
                activation = block['activation']    # 激活层，激活函数是用来加入非线性因素的，因为线性模型的表达能力不够。
                model = nn.Sequential() # 模型为空，通过add_module添加模型结构
                if batch_normalize:
                    # add_module(self, name, module)
                    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1,
                    #                                    padding=0, dilation=1, groups=1,
                    #                                     bias=True, padding_mode='zeros')
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    # print('create_network: conv{0}'.format(conv_id))
                    # 块标准化的目的就是让传输的数据合理的分布，加速训练的过程
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                else:
                    print(conv_id,"convalution havn't activate {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride == 1 and pool_size % 2:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=3 stride=1
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=2 stride=2
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    model = MaxPoolDark(pool_size, stride)
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(reduction='mean')
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(reduction='mean')
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(reduction='mean')
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)

                models.append(Upsample_expand(stride))
                # models.append(Upsample_interpolate(stride))

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block['groups'])
                        prev_stride = out_strides[layers[0]] // int(block['groups'])
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1 or layers[1] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + \
                                   out_filters[layers[3]]
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors) // loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                self.num_classes = yolo_layer.num_classes
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                yolo_layer.scale_x_y = float(block['scale_x_y'])
                # yolo_layer.object_scale = float(block['object_scale'])
                # yolo_layer.noobject_scale = float(block['noobject_scale'])
                # yolo_layer.class_scale = float(block['class_scale'])
                # yolo_layer.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    # def save_weights(self, outfile, cutoff=0):
    #     if cutoff <= 0:
    #         cutoff = len(self.blocks) - 1
    #
    #     fp = open(outfile, 'wb')
    #     self.header[3] = self.seen
    #     header = self.header
    #     header.numpy().tofile(fp)
    #
    #     ind = -1
    #     for blockId in range(1, cutoff + 1):
    #         ind = ind + 1
    #         block = self.blocks[blockId]
    #         if block['type'] == 'convolutional':
    #             model = self.models[ind]
    #             batch_normalize = int(block['batch_normalize'])
    #             if batch_normalize:
    #                 save_conv_bn(fp, model[0], model[1])
    #             else:
    #                 save_conv(fp, model[0])
    #         elif block['type'] == 'connected':
    #             model = self.models[ind]
    #             if block['activation'] != 'linear':
    #                 save_fc(fc, model)
    #             else:
    #                 save_fc(fc, model[0])
    #         elif block['type'] == 'maxpool':
    #             pass
    #         elif block['type'] == 'reorg':
    #             pass
    #         elif block['type'] == 'upsample':
    #             pass
    #         elif block['type'] == 'route':
    #             pass
    #         elif block['type'] == 'shortcut':
    #             pass
    #         elif block['type'] == 'region':
    #             pass
    #         elif block['type'] == 'yolo':
    #             pass
    #         elif block['type'] == 'avgpool':
    #             pass
    #         elif block['type'] == 'softmax':
    #             pass
    #         elif block['type'] == 'cost':
    #             pass
    #         else:
    #             print('unknown type %s' % (block['type']))
    #     fp.close()
