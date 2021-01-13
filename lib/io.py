import argparse, os, sys

import numpy as np
import torch
import torch.nn as nn

def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict

def save_ckpt(ckpt_name, models, optimizers, step, epoch):
    ckpt_dict = {'n_iter': epoch, 'step' : step}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)

def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name, map_location='cpu')
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])

    epoch = ckpt_dict['n_iter'] if 'n_iter' in ckpt_dict else 0
    step = ckpt_dict['step'] if 'step' in ckpt_dict else 0

    return step,  epoch

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()