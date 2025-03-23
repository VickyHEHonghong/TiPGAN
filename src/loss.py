import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg_models import GramMatrix, GramMSELoss, VGGModel


class HistogramLoss(nn.Module):
    EPS = 1e-6

    def __init__(self,
                 h=64,
                 insz=256,
                 resizing='interpolation',
                 method='inverse-quadratic',
                 sigma=0.02,
                 intensity_scale=True,
                 device='cuda'):
        """ Computes the RGB-uv histogram feature of a given image.
        Args:
            h: histogram dimension size (scalar). The default value is 64.
            insz: maximum size of the input image; if it is larger than this size, the
                image will be resized (scalar). Default value is 150 (i.e., 150 x 150
                pixels).
            resizing: resizing method if applicable. Options are: 'interpolation' or
                'sampling'. Default is 'interpolation'.
            method: the method used to count the number of pixels for each bin in the
                histogram feature. Options are: 'thresholding', 'RBF' (radial basis
                function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
            sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
                the sigma parameter of the kernel function. The default value is 0.02.
            intensity_scale: boolean variable to use the intensity scale (I_y in
                Equation 2). Default value is True.

        Methods:
            forward: accepts input image and returns its histogram feature. Note that
                unless the method is 'thresholding', this is a differentiable function
                and can be easily integrated with the loss function. As mentioned in the
                paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
                training.
        """
        super(HistogramLoss, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        if self.method == 'thresholding':
            self.eps = 6.0 / h
        else:
            self.sigma = sigma

    def forward_block(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x,
                                          size=(self.insz, self.insz),
                                          mode='bilinear',
                                          align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h,
                                endpoint=False)).to(device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h,
                                endpoint=False)).to(device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros(
            (x_sampled.shape[0], 3, self.h, self.h)).to(device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] +
                                                II[:, 2] + self.EPS),
                                     dim=1)
            else:
                Iy = 1

            Iu0 = torch.unsqueeze(torch.log(I[:, 0] + self.EPS) -
                                  torch.log(I[:, 1] + self.EPS),
                                  dim=1)
            Iv0 = torch.unsqueeze(torch.log(I[:, 0] + self.EPS) -
                                  torch.log(I[:, 2] + self.EPS),
                                  dim=1)
            diff_u0 = abs(
                Iu0 -
                torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                dim=0).to(self.device))
            diff_v0 = abs(
                Iv0 -
                torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                dim=0).to(self.device))
            if self.method == 'thresholding':
                diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                    2) / self.sigma**2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                    2) / self.sigma**2
                diff_u0 = torch.exp(-diff_u0)
            elif self.method == 'inverse-quadratic':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                    2) / self.sigma**2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                    2) / self.sigma**2
                diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                diff_v0 = 1 / (1 + diff_v0)
            else:
                raise Exception(
                    f'Wrong kernel method. It should be either thresholding, RBF,'
                    f' inverse-quadratic. But the given value is {self.method}.'
                )
            diff_u0 = diff_u0.type(torch.float32)
            diff_v0 = diff_v0.type(torch.float32)
            a = torch.t(Iy * diff_u0)
            hists[l, 0, :, :] = torch.mm(a, diff_v0)

            Iu1 = torch.unsqueeze(torch.log(I[:, 1] + self.EPS) -
                                  torch.log(I[:, 0] + self.EPS),
                                  dim=1)
            Iv1 = torch.unsqueeze(torch.log(I[:, 1] + self.EPS) -
                                  torch.log(I[:, 2] + self.EPS),
                                  dim=1)
            diff_u1 = abs(
                Iu1 -
                torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                dim=0).to(self.device))
            diff_v1 = abs(
                Iv1 -
                torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma**2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma**2
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == 'inverse-quadratic':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma**2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma**2
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

        diff_u1 = diff_u1.type(torch.float32)
        diff_v1 = diff_v1.type(torch.float32)
        a = torch.t(Iy * diff_u1)
        hists[l, 1, :, :] = torch.mm(a, diff_v1)

        Iu2 = torch.unsqueeze(torch.log(I[:, 2] + self.EPS) -
                              torch.log(I[:, 0] + self.EPS),
                              dim=1)
        Iv2 = torch.unsqueeze(torch.log(I[:, 2] + self.EPS) -
                              torch.log(I[:, 1] + self.EPS),
                              dim=1)
        diff_u2 = abs(
            Iu2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                  dim=0).to(self.device))
        diff_v2 = abs(
            Iv2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                  dim=0).to(self.device))
        if self.method == 'thresholding':
            diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
            diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
        elif self.method == 'RBF':
            diff_u2 = torch.pow(torch.reshape(diff_u2,
                                              (-1, self.h)), 2) / self.sigma**2
            diff_v2 = torch.pow(torch.reshape(diff_v2,
                                              (-1, self.h)), 2) / self.sigma**2
            diff_u2 = torch.exp(-diff_u2)  # Gaussian
            diff_v2 = torch.exp(-diff_v2)
        elif self.method == 'inverse-quadratic':
            diff_u2 = torch.pow(torch.reshape(diff_u2,
                                              (-1, self.h)), 2) / self.sigma**2
            diff_v2 = torch.pow(torch.reshape(diff_v2,
                                              (-1, self.h)), 2) / self.sigma**2
            diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
            diff_v2 = 1 / (1 + diff_v2)
        diff_u2 = diff_u2.type(torch.float32)
        diff_v2 = diff_v2.type(torch.float32)
        a = torch.t(Iy * diff_u2)
        hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / ((
            (hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) +
                                    self.EPS)

        return hists_normalized

    def forward(self, pred, gt) -> torch.Tensor:
        pred_hist = self.forward_block(pred)
        gt_hist = self.forward_block(gt)
        histogram_loss = (1 / np.sqrt(2.0) * (torch.sqrt(
            torch.sum(torch.pow(
                torch.sqrt(gt_hist) - torch.sqrt(pred_hist), 2)))) /
                          pred_hist.shape[0])
        return histogram_loss * 1e-5


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self,
                 use_lsgan=False,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(
                    self.real_label)
                #  = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(
                    self.fake_label)
                #  = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real) -> torch.Tensor:
        target_tensor = self.get_target_tensor(input, target_is_real)
        target_tensor = target_tensor.cuda()
        # print(target_tensor)
        return self.loss(input, target_tensor)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = VGGModel()
        self.vgg.load_state_dict(torch.load('ckpt/vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()

    def __call__(self, fake_B, real_B) -> torch.Tensor:
        style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        # self.content_layers = ['r42']
        loss_layers = style_layers
        loss_fns = [GramMSELoss()] * len(style_layers)
        if torch.cuda.is_available():
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        # vgg = VGGModel()
        # vgg.load_state_dict(torch.load(os.getcwd() + '/Models/' + 'vgg_conv.pth'))
        # self.vgg = torchvision.models.vgg19(pretrained=True)
        # for param in self.vgg.parameters():
        #     param.requires_grad = False
        # if torch.cuda.is_available():
        #     self.vgg.cuda()

        # print(vgg.state_dict().keys())
        style_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]
        # self.content_weights = [1e0]
        weights = style_weights
        # print(weights)
        style_targets = [
            GramMatrix()(A).detach() for A in self.vgg(real_B, style_layers)
        ]
        targets = style_targets
        out = self.vgg(fake_B, loss_layers)
        layer_losses = [
            weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)
        ]
        loss = sum(layer_losses)
        style_loss = loss.type(torch.FloatTensor)
        return style_loss
