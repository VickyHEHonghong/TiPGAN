import torch
import torch.nn as nn


class RelativeTV(nn.Module):
    REDUCTION = ['mean', 'sum', 'none']

    def __init__(self, eps=1e-5, reduction='mean', patch=8) -> None:
        super().__init__()
        assert reduction in self.REDUCTION
        self.eps = eps
        self.reduction = reduction
        self.patch = patch

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        assert len(data.shape) == 4, 'The input data must with a shape of ' \
            f'[B, C, W, H], but got {data.shape}'
        # tv before
        tv_b = self.tv_loss(data)

        _, _, w, h = data.shape
        data = data.repeat(1, 1, 2, 2)
        # center_crop
        data = data[:, :, w // 2:w // 2 + w, h // 2:h // 2 + h]
        # tv after
        tv_a = self.tv_loss(data)

        # relative tv loss
        loss = 0.5 * self.relative_tv_loss(tv_a, tv_b) + \
            0.5 * self.relative_tv_loss(tv_b, tv_a)
        return loss

    def relative_tv_loss(self, tv_a: torch.Tensor,
                         tv_b: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(1 - tv_a / (tv_b + self.eps))
        loss = torch.abs(1 - tv_a / (tv_b + self.eps))
        return loss

    def reshape(self, data: torch.Tensor) -> torch.Tensor:
        b, c, w, h = data.shape
        w_patched = w // self.patch
        h_patched = h // self.patch
        data = data.reshape(b, c, w_patched, self.patch, h_patched, self.patch)
        return data

    def tv_loss(self, img: torch.Tensor) -> torch.Tensor:
        img = self.reshape(img)
        diff1 = img[..., 1:, :, :, :] - img[..., :-1, :, :, :]
        diff2 = img[..., 1:, :] - img[..., :-1, :]

        # why abs instead of squre
        res1 = diff1.pow(2).mean([1, 2, 3, 4, 5])
        res2 = diff2.pow(2).mean([1, 2, 3, 4, 5])
        score = torch.sqrt(res1 + res2)

        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        else:
            return score
