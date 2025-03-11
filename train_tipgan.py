import logging
import os
import os.path
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import tqdm
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataset import HalfDataset
from src.loss import GANLoss, HistogramLoss, StyleLoss
from src.network import Discriminator, ResnetNetGenerator
from src.utils import tensor2img

data_root = 'experiments'
texture_name = 'nature_0001'
texture_root = os.path.join(data_root, texture_name)
os.makedirs(texture_root, exist_ok=True)


def config_logging(file_name: str,
                   console_level: int = logging.INFO,
                   file_level: int = logging.DEBUG):
    today = str(datetime.now()).replace(' ', '-').replace(':',
                                                          '_').split('.')[0]
    file_name = os.path.join(file_name, f'{today}.txt')
    file_handler = logging.FileHandler(file_name, mode='a', encoding='utf8')
    file_handler.setFormatter(
        logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S'))
    file_handler.setLevel(file_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S'))
    console_handler.setLevel(console_level)
    logging.basicConfig(level=min(console_level, file_level),
                        handlers=[file_handler, console_handler])


log_path = os.path.join(texture_root, 'tipgan')
os.makedirs(log_path, exist_ok=True)
config_logging(log_path)
logger = logging.getLogger('SeamlessGAN_new')

json_root = os.path.join(texture_root, 'vicky_split_labels')
img_root = os.path.join(texture_root, 'images')
ckpt_root = os.path.join(texture_root, 'ckpt')
ckpt_org = os.path.join(
    ckpt_root, 'seamlessgan_0.pth')  #----------------------------------------
texture_res_root = os.path.join(texture_root, 'result')
texture_res_compose_root = os.path.join(texture_root, 'result_compose')
os.makedirs(texture_res_root, exist_ok=True)
os.makedirs(texture_res_compose_root, exist_ok=True)

log_step = 20
vis_step = 50
validation_step = 10
img_path = 'media/nature_0001.jpg'
train_dataset = HalfDataset(img_path, fineSize=256, split_type='train')
data_loader = DataLoader(dataset=train_dataset,
                         batch_size=20,
                         shuffle=True,
                         num_workers=4,
                         pin_memory=True,
                         drop_last=False)
val_dataset = HalfDataset(img_path, fineSize=256, split_type='valid')  ## val
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=20,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

lambda_A = 1.0
lambda_B = 10.0
#3.创建网络模型Creat generator and discriminator
# Initialize generator and discriminator
generator = ResnetNetGenerator(input_nc=3,
                               output_nc=3,
                               ngf=64,
                               norm_layer=nn.InstanceNorm2d,
                               use_dropout=False,
                               n_blocks=6,
                               padding_type='reflect')
discriminator = Discriminator(input_nc=6, ndf=64, norm_layer=nn.InstanceNorm2d)

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()

#4.  define loss functions
criterionGAN = GANLoss(use_lsgan=False,
                       tensor=torch.FloatTensor,
                       target_real_label=1.0,
                       target_fake_label=0.0)
criterionL1 = torch.nn.L1Loss()
criterionStyle = StyleLoss()
criterionHistogram = HistogramLoss()

#5.优化器
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))

# resume-from
if ckpt_org is None:
    state_dict = torch.load(ckpt_org)
    generator.load_state_dict(state_dict['generator'])
    discriminator.load_state_dict(state_dict['discriminator'])
    optimizer_G.load_state_dict(state_dict['optim_G'])
    optimizer_D.load_state_dict(state_dict['optim_D'])
    begin_epoch = state_dict['epoch']
    current_train_step = begin_epoch * len(data_loader)
else:
    ##记录训练次数
    # begin_epoch = 2500
    # current_train_step = 700000
    begin_epoch = 0
    current_train_step = 0

#6.设置训练网络的一些参数
##添加tensorboard
writer = SummaryWriter(f'{texture_root}/logs_seamless_train_{texture_name}')
start_time = time.time()

##将tensor数据转换为img numpy array，方便可视化

# 训练的轮数
epoch = 5000
for i in range(begin_epoch, epoch):
    begin_time = time.perf_counter()
    for idx, batch_data in enumerate(data_loader):
        real_A_128 = batch_data['A']  ##real_A.size()=128*128
        real_B = batch_data['B']  ## real_B.size() = 256*256
        if torch.cuda.is_available():
            real_A_128 = real_A_128.cuda()
            real_B = real_B.cuda()
        data_time = time.perf_counter()

        # synthesizing fake images
        fake_B_512 = generator(real_A_128).cuda()
        fake_B = transforms.CenterCrop(256)(fake_B_512).cuda()
        real_A = transforms.RandomResizedCrop(256)(real_A_128)
        # training on discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        disc_fake = discriminator(
            fake_AB.detach())  ##disc_fake.size() = 1*1*64*64
        loss_D_fake = criterionGAN(disc_fake, target_is_real=False)

        real_AB = torch.cat((real_A, real_B), 1)  ##real_B.size() = 1*6*256*256
        disc_real = discriminator(real_AB)  ###disc_fake.size() = 1*1*64*64
        loss_D_real = criterionGAN(disc_real, target_is_real=True)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # training on generator
        fake_B_B256 = torch.cat((real_B, fake_B), 1)
        disc_fake = discriminator(fake_B_B256)

        loss_GAN = criterionGAN(disc_fake, True)
        loss_L1 = criterionL1(fake_B, real_B)
        loss_style = criterionStyle(fake_B, real_B)

        hist_loss = criterionHistogram(fake_B, real_B)

        loss_G = lambda_A * loss_L1 + lambda_B * loss_style + loss_GAN + hist_loss

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        current_train_step = current_train_step + 1
        if current_train_step % 10 == 0:
            end_time = time.time()
            print('time', end_time - start_time)
            # print({'loss_G', loss_G.item(),
            #        'Loss_GAN', loss_GAN.item(),
            #        'loss_L1', loss_L1.item(),
            #        'loss_style', loss_style.item()})

            print(
                '训练次数：{},loss_G:{},loss_D:{},Loss_GAN:{},loss_hist:{},loss_L1:{},loss_style:{}'
                .format(current_train_step, loss_G.item(), loss_D.item(),
                        loss_GAN.item(), hist_loss.item(), loss_L1.item(),
                        loss_style.item()))
            writer.add_scalar('seamless_train_loss/loss_G', loss_G.item(),
                              current_train_step)
            writer.add_scalar('seamless_train_loss/loss_D', loss_D.item(),
                              current_train_step)
            writer.add_scalar('seamless_train_loss/loss_GAN', loss_GAN.item(),
                              current_train_step)
            writer.add_scalar('seamless_train_loss/loss_hist',
                              hist_loss.item(), current_train_step)
            writer.add_scalar('seamless_train_loss/loss_L1', loss_L1.item(),
                              current_train_step)
            writer.add_scalar('seamless_train_loss/loss_style',
                              loss_style.item(), current_train_step)

        if current_train_step % 20 == 0:
            real_A_numpy = tensor2img(real_A_128)
            fake_B_numpy = tensor2img(fake_B)
            real_B_numpy = tensor2img(real_B)
            disc_fake = F.interpolate(disc_fake,
                                      scale_factor=(4, 4),
                                      mode='bilinear',
                                      align_corners=True)
            disc_fake_numpy = tensor2img(disc_fake)

            real_A_result = Image.fromarray(real_A_numpy)
            fake_B_result = Image.fromarray(fake_B_numpy)
            real_B_result = Image.fromarray(real_B_numpy)
            disc_fake_result = Image.fromarray(disc_fake_numpy)

            real_A_result.save(
                os.path.join(texture_res_root,
                             'real_A' + f'{current_train_step}.jpg'))
            fake_B_result.save(
                os.path.join(texture_res_root,
                             'fake_B' + f'{current_train_step}.jpg'))
            real_B_result.save(
                os.path.join(texture_res_root,
                             'real_B' + f'{current_train_step}.jpg'))
            disc_fake_result.save(
                os.path.join(texture_res_root,
                             'disc_fake' + f'{current_train_step}.jpg'))

    # if i % 500 ==0:
    #     ##保存模型
    #     model_dict = {
    #         'generator': generator.state_dict(),
    #         'discriminator': discriminator.state_dict(),
    #         'optim_G': optimizer_G.state_dict(),
    #         'optim_D': optimizer_D.state_dict(),
    #         'epoch': i
    #     }
    #     torch.save(model_dict, ckpt_root +  '/'+  'seamlessgan_{}.pth'.format(i))
