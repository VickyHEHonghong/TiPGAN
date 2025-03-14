import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import functional as F


def get_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)


class HalfDataset(data.Dataset):
    def __init__(self, img_path, fineSize=256, split_type='train'):
        super(HalfDataset, self).__init__()

        self.split_type = split_type
        self.paths = [img_path]
        self.size = len(self.paths)
        self.transform = get_transform()
        self.fineSize = fineSize

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        B_img = Image.open(path).convert('RGB')

        if self.split_type != 'train':
            w, h = B_img.size
            B_img = F.center_crop(B_img, 256)
            A_img = F.center_crop(B_img, 128)

            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        else:
            if B_img.size[1] > self.fineSize:
                w, h = B_img.size
                rw = random.randint(0, w - self.fineSize)
                rh = random.randint(0, h - self.fineSize)

                B_img = B_img.crop(
                    (rw, rh, rw + self.fineSize, rh + self.fineSize))

                w, h = B_img.size
                rw = random.randint(0, int(w / 2))
                rh = random.randint(0, int(h / 2))

                A_img = B_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))

                A_img = self.transform(A_img)
                B_img = self.transform(B_img)

            else:
                w, h = B_img.size
                rw = random.randint(0, abs(w - self.fineSize))
                rh = random.randint(0, abs(h - self.fineSize))

                B_img = B_img.resize((w + self.fineSize, h + self.fineSize))
                B_img = B_img.crop(
                    (rw, rh, rw + self.fineSize, rh + self.fineSize))

                w, h = B_img.size
                rw = random.randint(0, int(w / 2))
                rh = random.randint(0, int(h / 2))

                A_img = B_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))
                A_img = self.transform(A_img)
                B_img = self.transform(B_img)

        return {
            'A': A_img,
            'B': B_img,
            'A_size': A_img.size(),
            'B_size': B_img.size(),
            'A_paths': path,
            'B_paths': path
        }

    def __len__(self):
        return self.size

    def name(self):
        return 'HalfDataset'
