import sys
sys.path.append("..")

import torch
import torch.nn as nn

from lib.io import print_
from network.vgg import VGG16FeatureExtractor

def gram_matrix(feat):
    """
    Calculate gram matrix used in style loss
    https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    """
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)

    return gram

def normalize_batch(batch):
    """ Normalize batch using imagenet mean and std """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

class PerceptualLossBase(nn.Module):
    def __init__(self, extractor, device):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.device = device

    def _total_variation_loss(self, image, mask):
        """Total variation loss, used for smoothing the hole region"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        _, ch, _, _ = mask.shape
        dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(self.device)
        torch.nn.init.constant_(dilation_conv.weight, 1.0)
        with torch.no_grad():
            output_mask = dilation_conv(1-mask)

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp.
        dilated_holes = (output_mask != 0).float()
        P = dilated_holes*image

        # Calculate total variation loss.
        a = torch.mean(torch.abs(P[:, :, :, 1:]-P[:, :, :, :-1]))
        b = torch.mean(torch.abs(P[:, :, 1:, :]-P[:, :, :-1, :]))

        return a+b

    def forward(self, input, mask, output, gt):
        raise NotImplementedError()

class HDRLoss(PerceptualLossBase):
    def __init__(self, extractor, device):
        PerceptualLossBase.__init__(self, extractor, device)
        self.LAMBDA_DICT = { 'hole': 6.0, 'prc': 1.0, 'style' : 120 }

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        mask_hole = 1-mask
        output = torch.clamp(output, min=0, max=10)

        # Compute predicted image with non-hole pixels set to ground truth.
        log_gt = torch.log(gt+1)
        loss_dict['hole'] = self.l1(mask_hole*output, mask_hole*log_gt)

        # Other loss terms.
        with torch.no_grad():
            y = torch.exp(output)-1
            
        # Range compress images.
        y_clamp = torch.clamp(y, min=0, max=50)
        gt_clamp = torch.clamp(gt, min=0, max=50)
        
        k = gt_clamp.view(gt_clamp.shape[0],-1).max(1)[0].view((gt_clamp.shape[0],1,1,1))
        y_ = y_clamp / k
        gt_ = gt_clamp / k

        out_mu = self._mu_law(y_, 500)
        gt_mu = self._mu_law(gt_, 500)

        # Compose images.
        out_comp = mask*gt_mu + mask_hole*out_mu

        # Extract features maps.
        if output.shape[1] == 3:
            feat_output = self.extractor(normalize_batch(out_comp))
            feat_gt = self.extractor(normalize_batch(gt_mu))
        elif output.shape[1] == 1:
            feat_output = self.extractor(torch.cat([normalize_batch(out_comp)]*3, 1))
            feat_gt = self.extractor(torch.cat([normalize_batch(gt_mu)]*3, 1))
        else:
            raise ValueError('Data format error.')

        # Calculate VGG loss.
        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])

        # Calculate style loss.
        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))

        # Calculate total variation loss.
        if 'tv' in self.LAMBDA_DICT:
            loss_dict['tv'] = self._total_variation_loss(out_comp, mask)

        return loss_dict

    def _mu_law(self, H, mu=5000):
        x = torch.max(H, torch.tensor(0.0).to(self.device)).to(self.device)
        res = torch.log(1. + mu*x)/torch.log(torch.tensor(1.+mu))
        return res

class InpaintingLoss(PerceptualLossBase):
    def __init__(self, extractor, device):
        PerceptualLossBase.__init__(self, extractor, device)
        self.LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask*output, mask*gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('Data format error.')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        if 'tv' in self.LAMBDA_DICT:
            loss_dict['tv'] = self._total_variation_loss(output_comp, mask)

        return loss_dict

def load(mode, device = torch.device("cuda")):
    criterion = None
    if mode == "hdr":
        print_('\tExperiment running HDR loss.\n', bold=True)
        return HDRLoss(VGG16FeatureExtractor(), device).to(device)
    elif mode == "inpainting":
        print_('\tExperiment running inpainting loss.\n', bold=True)
        return InpaintingLoss(VGG16FeatureExtractor(), device).to(device)
    else:
        raise ValueError('unknown mode {}'.format(mode))