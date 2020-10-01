import glob
import os
import random
import sys
sys.path.append("..")

import numpy as np
import torch

from lib.img_io import load_training_pair
from lib.util import get_saturated_regions

class HDRDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, image_transform, train):
        super(HDRDataset, self).__init__()
        self.img_transform = image_transform
        self.is_train = train

        self.images, self.labels = self._load_dataset(images_dir, labels_dir)
        self.dataset_len = len(self.images)

    def __getitem__(self, index):
        input_dir, label_dir = self.images[index], self.labels[index]
        status, image, label = load_training_pair(label_dir, input_dir)

        if status:
            conv_mask = get_saturated_regions(image)
            conv_mask = 1-conv_mask
            label[label > 10000] = 10000
        
        image = self.img_transform(image)
        conv_mask = torch.from_numpy(conv_mask).permute(2,0,1)
        label = torch.from_numpy(label).permute(2,0,1)

        return image, conv_mask, label

    def __len__(self):
        return self.dataset_len

    def _load_dataset(self, images_dir, labels_dir):
        if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
            return [images_dir], [labels_dir]

        images = glob.glob(os.path.join(images_dir, '*'))
        labels = glob.glob(os.path.join(labels_dir, '*'))
        images.sort()
        labels.sort()
        
        assert len(images) == len(labels)
        for im , label in zip(images, labels):
            assert(im.split('/')[-1].split(".")[0] == label.split('/')[-1].split(".")[0])
        
        return images, labels