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
from torchvision import transforms, utils


class MyDataSet(data.Dataset):
    def __init__(self, age_min, age_max, image_dir, label_dir, output_size=(256, 256), training_set=True, obscure_age=True):
        self.image_dir = image_dir
        self.transform = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1, 1, 1])
        self.resize = transforms.Compose([
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])

        # load label file
        label = np.load(label_dir)
        train_len = int(0.95*len(label))
        self.training_set = training_set
        self.obscure_age = obscure_age
        if training_set:
            label = label[:train_len]
        else:
            label = label[train_len:]
        a_mask = np.zeros(len(label), dtype=bool)
        for i in range(len(label)):
            if int(label[i, 1]) in range(age_min, age_max): a_mask[i] = True
        self.label = label[a_mask]
        self.length = len(self.label)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = os.path.join(self.image_dir, self.label[index][0])
        if self.training_set and self.obscure_age:
            age_val = int(self.label[index][1]) + np.random.randint(-1, 1)
        else:
            age_val = int(self.label[index][1])
        age = torch.tensor(age_val)

        image = Image.open(img_name)
        img = self.resize(image)
        if img.size(0) == 1:
            img = torch.cat((img, img, img), dim = 0)
        img = self.transform(img)

        return img, age
