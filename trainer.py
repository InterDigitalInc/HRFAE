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
from torchvision import transforms, utils

from nets import *
from functions import *

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.config = config
        # Networks
        self.enc = Encoder()
        self.dec = Decoder()
        self.mlp_style = Mod_Net()
        self.dis = Dis_PatchGAN()
        self.classifier = VGG()
        # Optimizers
        self.gen_params = list(self.enc.parameters()) + list(self.dec.parameters()) + list(self.mlp_style.parameters())
        self.dis_params = list(self.dis.parameters())
        self.gen_opt = torch.optim.Adam(self.gen_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        self.dis_opt = torch.optim.Adam(self.dis_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        self.gen_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=config['step_size'], gamma=config['gamma'])
        self.dis_scheduler = torch.optim.lr_scheduler.StepLR(self.dis_opt, step_size=config['step_size'], gamma=config['gamma'])
        
    def initialize(self, vgg_dir):
        self.enc.apply(init_weights)
        self.dec.apply(init_weights)
        self.mlp_style.apply(init_weights)
        self.dis.apply(init_weights)
        vgg_state_dict = torch.load(vgg_dir)
        vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
        self.classifier.load_state_dict(vgg_state_dict)
    
    def dataparallel(self):
        self.enc = nn.DataParallel(self.enc)
        self.dec = nn.DataParallel(self.dec)
        self.dis = nn.DataParallel(self.dis)
        self.classifier = nn.DataParallel(self.classifier)
        print('Dataparallel models created!')

    def L1loss(self, input, target):
        return torch.mean(torch.abs(input - target))
    
    def L2loss(self, input, target):
        return torch.mean((input - target)**2)

    def CEloss(self, x, target_age):
        return nn.CrossEntropyLoss()(x, target_age)

    def GAN_loss(self, x, real=True):
        if real:
            target = torch.ones(x.size()).type_as(x)
        else:
            target = torch.zeros(x.size()).type_as(x)
        return nn.MSELoss(reduction='none')(x, target)

    def grad_penalty_r1(self, net, x, coeff=10):
        """Calculate R1 regularization gradient penalty"""
        x.requires_grad=True 
        real_predict = net(x)
        gradients = grad(outputs=real_predict.mean(), inputs=x, create_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (coeff/2) * ((gradients.norm(2, dim=1) ** 2).mean())
        return gradient_penalty
    
    def random_age(self, age_input, diff_val=20):
        age_output = age_input.clone()
        if diff_val > (self.config['age_max'] - self.config['age_min'])/2:
            diff_val = (self.config['age_max'] - self.config['age_min'])//2
        for i, age_ele in enumerate(age_output):
            if age_ele < self.config['age_min'] + diff_val:
                age_target = age_ele.clone().random_(age_ele + diff_val, self.config['age_max'])
            elif (self.config['age_min'] + diff_val) <= age_ele <= (self.config['age_max'] - diff_val):
                age_target = age_ele.clone().random_(self.config['age_min'] + 2*diff_val, self.config['age_max']+1)
                if age_target <= age_ele + diff_val:
                    age_target = age_target - 2*diff_val
            elif age_ele > self.config['age_max'] - diff_val:
                age_target = age_ele.clone().random_(self.config['age_min'], age_ele - diff_val)
            age_output[i] = age_target
        return age_output

    def gen_encode(self, x_a, age_a, age_b=0, training=False, target_age=0):
        if target_age:
            self.target_age = target_age
            age_modif = self.target_age*torch.ones(age_a.size()).type_as(age_a)
        else:
            age_modif = self.random_age(age_a, diff_val=25)

        # Generate modified image
        self.content_code_a, skip_1, skip_2 = self.enc(x_a)
        style_params_a = self.mlp_style(age_a)
        style_params_b = self.mlp_style(age_modif)
        
        x_a_recon = self.dec(self.content_code_a, style_params_a, skip_1, skip_2)
        x_a_modif = self.dec(self.content_code_a, style_params_b, skip_1, skip_2)
        
        return x_a_recon, x_a_modif, age_modif

    def compute_gen_loss(self, x_a, x_b, age_a, age_b, log=False):
        # Generate modified image
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a, age_b, training=True)
        
        # Feed into discriminator
        realism_a_modif = self.dis(x_a_modif)
        predict_age_pb = self.classifier(vgg_transform(x_a_modif))['fc8']
        
        # Get predicted age
        predict_age = get_predict_age(predict_age_pb)
        self.age_diff = torch.mean(torch.abs(predict_age - age_a_modif.float()))
        
        # Classification loss
        self.loss_class = self.CEloss(predict_age_pb, age_a_modif)
        
        # Reconstruction loss
        self.loss_recon = self.L1loss(x_a_recon, x_a)
        
        # Adversarial loss
        self.loss_adver = self.GAN_loss(realism_a_modif, True).mean()

        # Total Variation
        self.loss_tv = reg_loss(x_a_modif)

        self.loss_gen = self.config['w']['recon']*self.loss_recon + \
                        self.config['w']['class']*self.loss_class + \
                        self.config['w']['adver']*self.loss_adver + \
                        self.config['w']['tv']*self.loss_tv

        return self.loss_gen


    def compute_dis_loss(self, x_a, x_b, age_a, age_b):
        # Generate modified image
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a, age_b, training=True)

        self.realism_b = self.dis(x_b)
        self.realism_a_modif = self.dis(x_a_modif.detach())

        self.loss_gp = self.grad_penalty_r1(self.dis, x_b)  
        self.loss_dis = self.GAN_loss(self.realism_b, True).mean() + self.GAN_loss(self.realism_a_modif, False).mean()
        
        self.loss_dis_gp = self.config['w']['dis']*self.loss_dis + self.config['w']['gp']*self.loss_gp

        return self.loss_dis_gp
    
    
    def log_image(self, x_a, age_a, logger, n_epoch, n_iter):
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a)
        logger.log_images('epoch'+str(n_epoch+1)+'/iter'+str(n_iter+1)+'/content', clip_img(x_a), n_iter + 1)
        logger.log_images('epoch'+str(n_epoch+1)+'/iter'+str(n_iter+1)+'/content_recon'+str(age_a.cpu().numpy()[0]), clip_img(x_a_recon), n_iter + 1)
        logger.log_images('epoch'+str(n_epoch+1)+'/iter'+str(n_iter+1)+'/content_modif_'+str(age_a_modif.cpu().numpy()[0]), clip_img(x_a_modif), n_iter + 1)

    def log_loss(self, logger, n_iter):
        logger.log_value('loss/total', self.loss_gen.item() + self.loss_dis_gp.item(), n_iter + 1)
        logger.log_value('loss/recon', self.loss_recon.item(), n_iter + 1)
        logger.log_value('loss/class', self.loss_class.item(), n_iter + 1)
        logger.log_value('loss/adv', self.loss_adver.item(), n_iter + 1)
        logger.log_value('loss/dis', self.loss_dis_gp.item(), n_iter + 1)
        logger.log_value('age_diff', self.age_diff.item(), n_iter + 1) 
        logger.log_value('dis/realism_A_modif', self.realism_a_modif.mean().item(), n_iter + 1)
        logger.log_value('dis/realism_B', self.realism_b.mean().item(), n_iter + 1)
    
    def save_image(self, x_a, age_a, log_dir, n_epoch, n_iter):
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a)
        utils.save_image(clip_img(x_a), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +str(n_iter+1)+ '_content.png')
        utils.save_image(clip_img(x_a_recon), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +str(n_iter+1)+ '_content_recon_'+str(age_a.cpu().numpy()[0])+'.png')
        utils.save_image(clip_img(x_a_modif), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +str(n_iter+1)+ '_content_modif_'+str(age_a_modif.cpu().numpy()[0])+'.png')
    
    def test_eval(self, x_a, age_a, target_age=0, hist_trans=True):
        _, x_a_modif, _= self.gen_encode(x_a, age_a, target_age=target_age)
        if hist_trans:
            for j in range(x_a_modif.size(0)):
                x_a_modif[j] = hist_transform(x_a_modif[j], x_a[j])
        return x_a_modif
    

    def save_model(self, log_dir):
        torch.save(self.enc.state_dict(),'{:s}/enc.pth.tar'.format(log_dir))
        torch.save(self.mlp_style.state_dict(),'{:s}/mlp_style.pth.tar'.format(log_dir))
        torch.save(self.dec.state_dict(),'{:s}/dec.pth.tar'.format(log_dir))
        torch.save(self.dis.state_dict(),'{:s}/dis.pth.tar'.format(log_dir))

    def save_checkpoint(self, n_epoch, log_dir):
        checkpoint_state = {
            'n_epoch': n_epoch,
            'enc_state_dict': self.enc.state_dict(),
            'dec_state_dict': self.dec.state_dict(),
            'mlp_style_state_dict': self.mlp_style.state_dict(),
            'dis_state_dict': self.dis.state_dict(),
            'gen_opt_state_dict': self.gen_opt.state_dict(),
            'dis_opt_state_dict': self.dis_opt.state_dict(),
            'gen_scheduler_state_dict': self.gen_scheduler.state_dict(),
            'dis_scheduler_state_dict': self.dis_scheduler.state_dict()
        } 
        torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir))
        if (n_epoch+1) % 10 == 0 :
            torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir)+'_'+str(n_epoch+1))
    
    def load_model(self, log_dir):
        self.enc.load_state_dict(torch.load('{:s}/enc.pth.tar'.format(log_dir)))
        self.mlp_style.load_state_dict(torch.load('{:s}/mlp_style.pth.tar'.format(log_dir)))
        self.dis.load_state_dict(torch.load('{:s}/dis.pth.tar'.format(log_dir)))
        self.dec.load_state_dict(torch.load('{:s}/dec.pth.tar'.format(log_dir)))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.enc.load_state_dict(state_dict['enc_state_dict'])
        self.dec.load_state_dict(state_dict['dec_state_dict'])
        self.mlp_style.load_state_dict(state_dict['mlp_style_state_dict'])
        self.dis.load_state_dict(state_dict['dis_state_dict'])
        self.gen_opt.load_state_dict(state_dict['gen_opt_state_dict'])
        self.dis_opt.load_state_dict(state_dict['dis_opt_state_dict'])
        self.gen_scheduler.load_state_dict(state_dict['gen_scheduler_state_dict'])
        self.dis_scheduler.load_state_dict(state_dict['dis_scheduler_state_dict'])
        return state_dict['n_epoch'] + 1

    def update(self, x_a, x_b, age_a, age_b, n_iter):
        self.n_iter = n_iter
        self.dis_opt.zero_grad()
        self.compute_dis_loss(x_a, x_b, age_a, age_b).backward()
        self.dis_opt.step()
        self.gen_opt.zero_grad()
        self.compute_gen_loss(x_a, x_b, age_a, age_b).backward()
        self.gen_opt.step()

        
