import argparse
import datetime
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from src.dataset import HalfDataset
from src.loss import GANLoss, HistogramLoss, StyleLoss
from src.network import Discriminator, TiPGANGenerator
from src.utils import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='TiPGAN train')

    # Adding arguments
    parser.add_argument('--img_path',
                        type=str,
                        default='media/nature_0005.jpg',
                        help='Path to the input image')
    parser.add_argument('--save_base',
                        type=str,
                        default='experiments',
                        help='Base directory to save the results')
    parser.add_argument('--save_meta',
                        type=str,
                        default=None,
                        help='Meta name of log dir')
    parser.add_argument('--resume_from',
                        type=str,
                        default=None,
                        help='Path to the checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--log_step',
                        type=int,
                        default=20,
                        help='Step interval for logging')
    parser.add_argument('--vis_step',
                        type=int,
                        default=1000,
                        help='Step interval for visualization')
    parser.add_argument('--save_step',
                        type=int,
                        default=1000,
                        help='Step interval for saving the model')
    parser.add_argument('--lambda_hist',
                        type=float,
                        default=1e-4,
                        help='Weight for histogram loss')
    parser.add_argument('--lambda_style',
                        type=float,
                        default=10.0,
                        help='Weight for style loss')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0002,
                        help='Learning rate')
    parser.add_argument('--total_iter',
                        type=int,
                        default=50000,
                        help='Total number of training iterations')

    # Parse the arguments
    return parser.parse_args()


def train(args):
    img_path = args.img_path
    save_base = args.save_base
    resume_from = args.resume_from
    log_step = args.log_step
    vis_step = args.vis_step
    save_step = args.save_step
    lambda_hist = args.lambda_hist
    lambda_style = args.lambda_style
    learning_rate = args.learning_rate
    total_iter = args.total_iter

    texture_name = osp.splitext(osp.basename(img_path))[0]
    exp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save_meta is not None:
        exp_name = f'{args.save_meta}-{exp_name}'
    save_dir = osp.join(save_base, texture_name, exp_name)
    tensorboard_dir = osp.join(save_dir, 'tensorboard')
    save_img_dir = osp.join(save_dir, 'imgs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # create dataset
    train_dataset = HalfDataset(img_path, fineSize=256, split_type='train')
    data_loader = DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             persistent_workers=True,
                             pin_memory=True,
                             drop_last=False)

    # init networks
    generator = TiPGANGenerator(input_nc=3,
                                output_nc=3,
                                ngf=64,
                                norm_layer=nn.InstanceNorm2d,
                                use_dropout=False,
                                padding_type='reflect')
    discriminator = Discriminator(input_nc=6,
                                  ndf=64,
                                  norm_layer=nn.InstanceNorm2d)

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # loss functions
    criterionGAN = GANLoss(use_lsgan=False,
                           tensor=torch.FloatTensor,
                           target_real_label=1.0,
                           target_fake_label=0.0)
    criterionL1 = torch.nn.L1Loss()
    criterionStyle = StyleLoss()
    criterionHistogram = HistogramLoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=learning_rate,
                                   betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=learning_rate,
                                   betas=(0.5, 0.999))
    scheduler_G = MultiStepLR(optimizer_G,
                              milestones=[30000, 40000],
                              gamma=0.2)
    scheduler_D = MultiStepLR(optimizer_D,
                              milestones=[30000, 40000],
                              gamma=0.2)

    # resume-from
    if resume_from is not None:
        assert osp.isfile(resume_from) and resume_from.endswith('.pth')
        state_dict = torch.load(resume_from)
        generator.load_state_dict(state_dict['generator'])
        discriminator.load_state_dict(state_dict['discriminator'])
        optimizer_G.load_state_dict(state_dict['optim_G'])
        optimizer_D.load_state_dict(state_dict['optim_D'])
        current_iter = state_dict['current_iter']
    else:
        current_iter = 0

    writer = SummaryWriter(tensorboard_dir)

    progress_bar = tqdm(
        range(0, total_iter),
        initial=current_iter,
        desc='Steps',
    )

    epoch_nums = (total_iter - current_iter) // len(data_loader)
    for _ in range(epoch_nums):
        for _, batch_data in enumerate(data_loader):
            real_A_128 = batch_data['A']
            real_B = batch_data['B']
            if torch.cuda.is_available():
                real_A_128 = real_A_128.cuda()
                real_B = real_B.cuda()

            fake_B = generator(real_A_128)
            real_A = transforms.RandomResizedCrop(256)(real_A_128)

            # optimize discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            disc_fake = discriminator(fake_AB.detach())
            loss_D_fake = criterionGAN(disc_fake, target_is_real=False)

            real_AB = torch.cat((real_A, real_B), 1)
            disc_real = discriminator(real_AB)
            loss_D_real = criterionGAN(disc_real, target_is_real=True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # optimize generator
            fake_B_B256 = torch.cat((real_B, fake_B), 1)
            disc_fake = discriminator(fake_B_B256)

            loss_GAN = criterionGAN(disc_fake, True)
            loss_L1 = criterionL1(fake_B, real_B)
            loss_style = lambda_style * criterionStyle(fake_B, real_B)
            loss_hist = lambda_hist * criterionHistogram(fake_B, real_B)
            loss_G = loss_L1 + loss_style + loss_GAN + loss_hist

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            scheduler_G.step()
            scheduler_D.step()

            logs = dict(step=current_iter,
                        loss_D=loss_D,
                        loss_G=loss_G,
                        loss_L1=loss_L1,
                        loss_style=loss_style,
                        loss_GAN=loss_GAN,
                        loss_hist=loss_hist)

            current_iter = current_iter + 1
            if current_iter % log_step == 0:
                for key, value in logs.items():
                    if key == 'step':
                        continue
                    writer.add_scalar(f'TiPGAN_train/{key}', value.item(),
                                      logs['step'])

            if current_iter % vis_step == 0:
                real_A_result = tensor2img(real_A_128)
                fake_B_result = tensor2img(fake_B)
                real_B_result = tensor2img(real_B)
                real_A_result = cv2.resize(real_A_result,
                                           real_B_result.shape[:2])
                vis_result = np.concatenate(
                    (real_A_result, real_B_result, fake_B_result), axis=1)
                writer.add_image('TiPGAN_train/vis',
                                 vis_result.transpose(2, 0, 1), logs['step'])
                vis_result = Image.fromarray(vis_result)
                vis_result.save(osp.join(save_img_dir, f'{current_iter}.jpg'))

            if current_iter % save_step == 0:
                model_dict = {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optim_G': optimizer_G.state_dict(),
                    'optim_D': optimizer_D.state_dict(),
                    'current_iter': current_iter
                }
                torch.save(
                    model_dict,
                    osp.join(save_dir,
                             'checkpoint-iter_{}.pth'.format(current_iter)))

            # format logs for printing
            for key, value in logs.items():
                if torch.is_tensor(value):
                    logs[key] = f'{value.item():.2e}'

            # update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

    progress_bar.close()
    writer.close()
    print('Training finished!')


if __name__ == '__main__':
    args = parse_args()
    train(args)
