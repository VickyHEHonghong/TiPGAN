import functools
import random

import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


def divide_batched_tensor(tensor, line1_position, line2_position):
    _, _, tensor_height, tensor_width = tensor.shape
    if (line1_position <= 0 or line1_position >= tensor_width
            or line2_position <= 0 or line2_position >= tensor_height):
        print('Error: Invalid line positions.')
        return

    # Divide the batched tensor into four parts
    part1 = tensor[:, :, :line2_position, :line1_position]
    part2 = tensor[:, :, :line2_position, line1_position:]
    part3 = tensor[:, :, line2_position:, :line1_position]
    part4 = tensor[:, :, line2_position:, line1_position:]
    x1 = part4
    x2 = part3
    x3 = part2
    x4 = part1

    top_row = torch.cat((x1, x2), dim=3)
    bottom_row = torch.cat((x3, x4), dim=3)
    stitched = torch.cat((top_row, bottom_row), dim=2)

    return stitched


class ResidualBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
                         use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetNetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect'):
        super(ResnetNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if padding_type == 'reflect':
            encoder = [nn.ReflectionPad2d(padding=3)]
        elif padding_type == 'replicate':
            encoder = [nn.ReplicationPad2d(3)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        encoder = []

        encoder += [
            nn.ReplicationPad2d(padding=3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        encoder += [
            nn.Conv2d(ngf,
                      ngf * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        ]

        encoder += [
            nn.Conv2d(ngf * 2,
                      ngf * 4,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        ]

        bottle_neck = []
        for _ in range(n_blocks):
            bottle_neck += [
                ResidualBlock(ngf * 4, padding_type, norm_layer, use_dropout,
                              use_bias)
            ]

        bottle_neck += [
            nn.Conv2d(ngf * 4,
                      ngf * 8,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        ]

        decoder = []
        decoder += [
            nn.ConvTranspose2d(ngf * 8,
                               int(ngf * 4),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias),
            norm_layer(int(ngf * 2)),
            nn.ReLU(True)
        ]

        decoder += [
            nn.ConvTranspose2d(ngf * 4,
                               int(ngf * 2),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias),
            norm_layer(int(ngf)),
            nn.ReLU(True)
        ]

        decoder += [
            nn.ConvTranspose2d(ngf * 2,
                               int(ngf),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias),
            norm_layer(int(ngf)),
            nn.ReLU(True)
        ]

        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]

        self.encoder = nn.Sequential(*encoder)
        self.bottle_neck = nn.Sequential(*bottle_neck)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        x0 = self.encoder(input)
        _, _, height, width = x0.size()
        rw = int(width / 4)
        rh = int(height / 4)
        rw_position = random.randint(rw + 2, width - (rw + 2))
        rh_position = random.randint(rh + 2, height - (rh + 2))
        d0 = divide_batched_tensor(x0, rw_position, rh_position)

        feature0 = d0.repeat(1, 1, 2, 2)
        seamless_feature = self.bottle_neck(feature0)
        seamless = self.decoder(seamless_feature)

        return seamless


class Discriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf,
                      ndf * 2,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 2,
                      ndf * 4,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * 4,
                      ndf * 8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=use_bias),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(int(ndf * 8), 1, kernel_size=4, stride=1, padding=2),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PatchSwapModule(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, n_blocks=6):
        super().__init__()

        blocks = []
        for _ in range(n_blocks):
            blocks += [
                ResidualBlock(dim, padding_type, norm_layer, use_dropout,
                              use_bias)
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor):
        _, _, height, width = inputs.size()
        rw = int(width / 4)
        rh = int(height / 4)
        if self.training:
            rw_position = random.randint(rw + 2, width - (rw + 2))
            rh_position = random.randint(rh + 2, height - (rh + 2))
        else:
            rw_position = rw + 2
            rh_position = rh + 2
        d0 = divide_batched_tensor(inputs, rw_position, rh_position)
        features = self.blocks(d0)
        return features


class PatchTilingModule(nn.Module):
    def __init__(self, ngf, norm_layer=nn.InstanceNorm2d, use_bias=True):
        super().__init__()
        blocks = [nn.Conv2d(ngf * 4,
                      ngf * 8,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_bias),
                norm_layer(ngf * 8),
                nn.ReLU(True)
                ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor):
        features = inputs.repeat(1, 1, 2, 2)
        features = self.blocks(features)
        return features


class TiPGANGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 padding_type='reflect'):
        super(TiPGANGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if padding_type == 'reflect':
            encoder = [nn.ReflectionPad2d(padding=3)]
        elif padding_type == 'replicate':
            encoder = [nn.ReplicationPad2d(3)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        encoder = []

        encoder += [
            nn.ReplicationPad2d(padding=3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        encoder += [
            nn.Conv2d(ngf,
                      ngf * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        ]

        encoder += [
            nn.Conv2d(ngf * 2,
                      ngf * 4,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        ]

        self.psm = PatchSwapModule(4*ngf, padding_type, norm_layer, use_dropout, use_bias, n_blocks=6)
        
        self.ptm = PatchTilingModule(ngf, norm_layer, use_bias)
        
        decoder = []
        decoder += [
            nn.ConvTranspose2d(ngf * 8,
                               int(ngf * 4),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias),
            norm_layer(int(ngf * 2)),
            nn.ReLU(True)
        ]

        decoder += [
            nn.ConvTranspose2d(ngf * 4,
                               int(ngf * 2),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias),
            norm_layer(int(ngf)),
            nn.ReLU(True)
        ]

        decoder += [
            nn.ConvTranspose2d(ngf * 2,
                               int(ngf),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=use_bias),
            norm_layer(int(ngf)),
            nn.ReLU(True)
        ]

        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        self.center_crop = CenterCrop(256)
        
    def forward(self, inputs):
        feature = self.encoder(inputs)
        feature_psm = self.psm(feature)
        feature_ptm = self.ptm(feature_psm)
        seamless_texture = self.decoder(feature_ptm)
        seamless_texture = self.center_crop(seamless_texture)
        return seamless_texture
