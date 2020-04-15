# Copyright (c) 2020, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

print('By installing and using this software, you agree to comply with the license terms provided on the LICENSE.txt.')

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from torchvision import transforms, utils

from datasets import *
from nets import *
from functions import *
from trainer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='params', help='path to the config file.')
parser.add_argument('--vgg_model_path', type=str, default='./models/dex_imdb_wiki.caffemodel.pt', help='pretrained age classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--img_path', type=str, default='./test/input/', help='test image path')
parser.add_argument('--out_path', type=str, default='./test/output/', help='test output path')
parser.add_argument('--target_age', type=int, default=65, help='Age transform target, interger value between 20 and 70')
opts = parser.parse_args()

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(opts.out_path):
    os.makedirs(opts.out_path)

config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'))
img_size = (config['input_w'], config['input_h'])

# Initialize trainer
trainer = Trainer(config)

# Load pretrained model 
if opts.checkpoint:
    trainer.load_checkpoint(opts.checkpoint)
else:
    trainer.load_checkpoint(log_dir + 'checkpoint')

trainer.to(device)

# Set target age
target_age = opts.target_age

# Load test image
img_list = os.listdir(opts.img_path)
img_list.sort()

# Preprocess
def preprocess(img_name):
    resize = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            ])
    normalize = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1,1,1])
    img_pil = Image.open(opts.img_path + img_name)
    img_np = np.array(img_pil)
    img = resize(img_pil)
    if img.size(0) == 1:
        img = torch.cat((img, img, img), dim = 0)
    img = normalize(img)
    return img

with torch.no_grad():
    for img_name in img_list:
        if not img_name.endswith(('png', 'jpg', 'PNG', 'JPG')):
            print('File ignored: ' + img_name)
            continue
        image_A = preprocess(img_name)
        image_A = image_A.unsqueeze(0).to(device)

        age_modif = torch.tensor(target_age).unsqueeze(0).to(device)
        image_A_modif = trainer.test_eval(image_A, age_modif, target_age=target_age, hist_trans=True)  
        utils.save_image(clip_img(image_A_modif), opts.out_path + img_name.split('.')[0] + '_age_' + str(target_age) + '.jpg')

