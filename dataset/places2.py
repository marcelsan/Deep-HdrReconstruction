import glob
import os
import random
import sys
sys.path.append("..")

import torch

from PIL import Image
from lib.util import random_mask

class Places2(torch.utils.data.Dataset):
    def __init__(self, images_dir, image_transform, train):
        super(Places2, self).__init__()
        self.images = self._load_dataset(images_dir)
        self.dataset_len = len(self.images)
        self.img_transform = image_transform
        self.is_train = train

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.img_transform(image.convert('RGB'))
        mask = random_mask(image.shape[1], image.shape[2])
        mask = torch.from_numpy(mask).permute(2,0,1)
        input_image = image * mask
        input_image[mask==0] = 1

        return input_image, mask, image

    def __len__(self):
        return self.dataset_len

    def _load_dataset(self, images_dir):
        if not os.path.isdir(images_dir):
            return [images_dir]

        files = []
        for p, d, f in os.walk(images_dir):
            for file in f:
                if file.endswith('.jpg') or file.endswith('.png'):
                    files.append(os.path.join(p, file))

        return files