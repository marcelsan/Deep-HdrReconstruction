import os
import sys
sys.path.append("..")

from torchvision import transforms

import opt

from dataset.hdrdata import HDRDataset
from dataset.places2 import Places2
from lib.io import print_
from lib.util import get_data_directory

def load(args):
	img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

	if args.mode == "hdr":
		print_('\tExperiment running on HDR data.\n', bold=True)

		if args.train:
			dataset_train = HDRDataset(os.path.join(args.train_dir,'jpg'), os.path.join(args.train_dir,'bin'), img_transform, train=True)
			dataset_val = HDRDataset(os.path.join(args.val_dir,'jpg'), os.path.join(args.val_dir,'bin'), img_transform, train=False)
			return dataset_train, dataset_val
		else:
			im_dir, label_dir = get_data_directory(args.test_dir)
			return HDRDataset(im_dir, label_dir, img_transform, train=False)

	elif args.mode == "inpainting":
		print_('\tExperiment running on inpainting data.\n', bold=True)

		if args.train:
			dataset_train = Places2(args.train_dir, img_transform, True)
			dataset_val = Places2(args.val_dir, img_transform, False)
			return dataset_train, dataset_val
		else:
			return Places2(args.test_dir, img_transform, train=False)
			
	else:
		raise ValueError('Unknown mode {}. Choose either hdr or inpainting.'.format(args.mode))