import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from network.net_utils import weights_init

class SoftConvNotLearnedMask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, 1, bias)
        self.mask_update_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, 1, False)
        self.input_conv.apply(weights_init('xavier'))
        
    def forward(self, input, mask):
        output = self.input_conv(input * mask)

        with torch.no_grad():
            self.mask_update_conv.weight = torch.nn.Parameter(self.input_conv.weight.abs())
            filters, _, _, _ = self.mask_update_conv.weight.shape
            k = self.mask_update_conv.weight.view((filters, -1)).sum(1)
            norm = k.view(1,-1,1,1).repeat(mask.shape[0],1,1,1)
            new_mask = self.mask_update_conv(mask)/(norm + 1e-6) 

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu', conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 5, 2, 2, bias=conv_bias) # Downsampling by 2
        elif sample == 'down-7':
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 7, 2, 3, bias=conv_bias) # Downsampling by 2
        elif sample == 'down-3':
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 3, 2, 1, bias=conv_bias) # Downsampling by 2
        else:
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
            
    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        
        if hasattr(self, 'activation'):
            h = self.activation(h)
            
        return h, h_mask

class SoftConvNotLearnedMaskUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i+1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512+512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512+256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256+128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128+64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64+input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')
            
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, torch.ones_like(h_mask_dict[enc_h_key])], dim=1)
        
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            h_mask_dict[dec_l_key] = h_mask

        return h

    def train(self, mode=True):
        """ Override the default train() to freeze the BN parameters """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()
