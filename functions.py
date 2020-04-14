# Copyright (c) 2020, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torch.autograd import grad

        
def clip_img(x):
    """Clip image to range(0,1)"""
    img_tmp = x.clone()[0]
    img_tmp[0] += 0.48501961
    img_tmp[1] += 0.45795686
    img_tmp[2] += 0.40760392
    img_tmp = torch.clamp(img_tmp, 0, 1)
    return [img_tmp.detach().cpu()]
    
def hist_transform(source_tensor, target_tensor):
    """Histogram transformation"""
    c, h, w = source_tensor.size()
    s_t = source_tensor.view(c, -1)
    t_t = target_tensor.view(c, -1)
    s_t_sorted, s_t_indices = torch.sort(s_t)
    t_t_sorted, t_t_indices = torch.sort(t_t)
    for i in range(c):
        s_t[i, s_t_indices[i]] = t_t_sorted[i]
    return s_t.view(c, h, w)

def init_weights(m):
    """Initialize layers with Xavier uniform distribution"""
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01)
    elif type(m) == nn.Linear:
        nn.init.uniform_(m.weight, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


def reg_loss(img):
    """Total variation"""
    reg_loss = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))\
             + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return reg_loss

def vgg_transform(x):
    """Adapt image for vgg network, x: image of range(0,1) subtracting ImageNet mean"""
    r, g, b = torch.split(x, 1, 1)
    out = torch.cat((b, g, r), dim = 1)
    out = F.interpolate(out, size=(224, 224), mode='bilinear')
    out = out*255.
    return out

def get_predict_age(age_pb):
    predict_age_pb = F.softmax(age_pb)
    predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
    for i in range(age_pb.size(0)):
        for j in range(age_pb.size(1)):
            predict_age[i] += j*predict_age_pb[i][j]
    return predict_age

