import argparse
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.network import TiPGANGenerator
from src.dataset import HalfDataset


def parse_args():
    parser = argparse.ArgumentParser(description='TiPGAN Eval')

    # Adding arguments
    parser.add_argument('--img_path',
                        type=str,
                        default=None,
                        help='Path to the input image')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default=None,
                        help='Path to the checkpoint')

    # Parse the arguments
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_path = args.img_path
    ckpt_path = args.ckpt_path
    
    img_name = osp.splitext(osp.basename(img_path))[0]
    
    ckpt = torch.load(ckpt_path)
    dataset = HalfDataset(img_path=img_path)
    dataloader = DataLoader(dataset, batch_size=1)
    generator = TiPGANGenerator(input_nc=3,
                                output_nc=3,
                                ngf=64,
                                norm_layer=nn.InstanceNorm2d,
                                use_dropout=False,
                                padding_type='reflect')
    generator.load_state_dict(ckpt['generator'])
    del ckpt
    generator = generator.cuda().eval()

    for batch in dataloader:
        with torch.no_grad():
            # make sure to input an image with size (128, 128)
            # value range from (-1, 1)
            img = generator(batch['A'].cuda())
    tilled = img.repeat(1, 1, 2, 2)
    save_image(img, f'seamless-{img_name}.jpg', normalize=True, value_range=(-1, 1))
    save_image(tilled, f'seamless_tilled-{img_name}.jpg', normalize=True, value_range=(-1, 1))
