import argparse
import glob
import numpy as np
import os
import torch

from torchvision import transforms
from torch.utils import data

import opt

from lib.image import unnormalize
from lib.img_io import load_image, writeLDR, writeEXR
from lib.io import load_ckpt
from lib.io import print_
from lib.util import make_dirs
from lib.util import get_saturated_regions
from network.softconvmask import SoftConvNotLearnedMaskUNet

parser = argparse.ArgumentParser(description="Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss")
parser.add_argument('--test_dir', '-t', type=str, required=True, help='Input images directory.')
parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to output directory.')
parser.add_argument('--weights', '-w', type=str, required=True, help='Path to the trained CNN weights.')
parser.add_argument('--cpu', action='store_true')

def print_test_args(args):
    print_("\n\n\t-------------------------------------------------------------------\n", 'm')
    print_("\t  HDR image reconstruction from a single exposure using deep CNNs\n\n", 'm')
    print_("\t  Settings\n", 'm')
    print_("\t  -------------------\n", 'm')
    print_("\t  Input image directory/file:     %s\n" % args.test_dir, 'm')
    print_("\t  Output directory:               %s\n" % args.out_dir, 'm')
    print_("\t  CNN weights:                    %s\n" % args.weights, 'm')
    print_("\t-------------------------------------------------------------------\n\n\n", 'm')

class HDRTestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, image_transform):
        super(HDRTestDataset, self).__init__()
        self.images = self._load_dataset(images_dir)
        self.dataset_len = len(self.images)
        self.img_transform = image_transform

    def __getitem__(self, index):
        input_dir = self.images[index]

        # load image
        image = load_image(input_dir)

        # get saturation mask
        conv_mask = 1 - get_saturated_regions(image)
        conv_mask = torch.from_numpy(conv_mask).permute(2,0,1)

        # apply transform to input image
        image = self.img_transform(image)

        return image, conv_mask

    def __len__(self):
        return self.dataset_len

    def _load_dataset(self, images_dir):
        images = []
        for ext in ('*.png', '*.jpeg', '*.jpg'):
           images.extend(glob.glob(os.path.join(images_dir, ext)))
        images.sort()
        
        return images

if __name__ == '__main__':
    args = parser.parse_args()
    args.train = False
    print_test_args(args)

    # use GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print_('\tUsing device: {}.\n'.format(device))

    # create output directory.
    make_dirs(args.out_dir)

    # load test data. 
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

    dataset = HDRTestDataset(args.test_dir, img_transform)
    iterator_test_set = data.DataLoader(dataset, batch_size=1)
    print_('\tLoaded {} test images.\n'.format(len(dataset)))

    model = SoftConvNotLearnedMaskUNet().to(device)
    load_ckpt(args.weights, [('model', model)])

    print_("Starting prediction...\n\n")
    model.eval()
    for i, (image, mask) in enumerate(iterator_test_set):
        print("Image %d/%d"%(i+1, len(dataset)))
        print_("\t(Saturation: %0.2f%%)\n" % (100.0*(1-mask.mean().item())), 'm')
        print_("\tInference...\n")

        with torch.no_grad():
            pred_img = model(image.to(device), mask.to(device))

        print_("\tdone...\n")

        image = unnormalize(image).permute(0,2,3,1).numpy()[0,:,:,:]
        mask = mask.permute(0,2,3,1).numpy()[0,:,:,:]
        pred_img = pred_img.cpu().permute(0,2,3,1).numpy()[0,:,:,:]

        y_predict = np.exp(pred_img)-1
        gamma = np.power(image, 2)

        # save EXR images.
        H = mask*gamma + (1-mask)*y_predict

        # write images to disc.
        writeLDR(mask, '%s/%d_mask.png' % (args.out_dir, i+1))
        writeEXR(H, '{}/img_{}.exr'.format(args.out_dir, i+1))
        writeEXR(gamma, '{}/img_gamma_{}.exr'.format(args.out_dir, i+1))

    print_('done!! \n')