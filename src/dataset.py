import json
import os
import os.path
import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms.functional import crop


def split_image_into_patches(image):
    height, width = image.shape[-2], image.shape[-1]
    patch_height = height // 2
    patch_width = width // 2
    patches = []
    for i in range(2):
        for j in range(2):
            patch = crop(image, i * patch_height, j * patch_width,
                         patch_height, patch_width)
            patches.append(patch)
    return patches


def make_dataset(json_root, img_root, split_type):
    images = []
    assert split_type in ['train', 'test', 'valid']
    json_path = os.path.join(json_root, f'{split_type}.json')
    # get the image paths of your dataset;
    with open(json_path, 'r') as fp:
        image_names = json.load(fp)
        train_image_names = image_names[f'{split_type}_list']
        for img_name in train_image_names:
            img_path = os.path.join(img_root, img_name)
            images.append(img_path)
        print(images)
        return images


def get_transform():
    transform_list = []
    # resize_or_crop = 'resize_and_crop'
    # loadSize = 286
    # osize = [loadSize, loadSize]
    # transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    # transform_list.append(transforms.RandomCrop(fineSize))

    # # if opt.isTrain and not opt.no_flip:
    # #     transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [
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

        # TODO
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
