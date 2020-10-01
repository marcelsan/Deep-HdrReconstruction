# Deep-HdrReconstruction
Official PyTorch implementation of "Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss" (SIGGRAPH 2020) [Project](https://people.engr.tamu.edu/nimak/Papers/SIGGRAPH2020_HDR) | [Paper](https://people.engr.tamu.edu/nimak/Data/SIGGRAPH20_HDR.pdf)

We propose a novel deep learning system for single image HDR reconstruction by synthesizing visually pleasing details in the saturated areas. For more information on the method please see the [project website](https://people.engr.tamu.edu/nimak/Papers/SIGGRAPH2020_HDR).

![image](https://people.engr.tamu.edu/nimak/Papers/SIGGRAPH2020_HDR/files/teaser.png)

## Requirements
This codebase was developed and tested with PyTorch 1.2 and Python 3.6.

- Python 3.6+
- Pytorch 1.2
- torchvision
- opencv
- numpy
- tensorboardX
- tqdm
- Pillow
- pyexr

```
pip install -r requirements.txt
```

You may have to install OpenEXR through the appropriate package manager before pip install (e.g. sudo apt-get install openexr and libopenexr-dev on Ubuntu).

Download the repository

```
https://github.com/marcelsan/Deep-HdrReconstruction.git
```

## Usage

### Pretrained model

The pretrained model checkpoints can be found in the checkpoints folder on [Google Drive]().

### Inference

```
CUDA_VISIBLE_DEVICES=1 python test_hdr.py --test_dir <images/dir> --out_dir <out/dir> --weights <weight/path>.pth 

```

Parameters and their description:

>```test_dir```: input images directory. A few images are avaible on the data/ folder.<br/>
>```out_dir```: path to output directory.<br/>
>```weights```: path to the trained CNN weights.<br/>
<br/>

If cuda is available, it will be used. In case you want to run the model on cpu, use ```cpu``` when executing test_hdr.py

## References
If you find this work useful for your research, please cite:

```
@article{Marcel:2020:LDRHDR,
author = {Santos, Marcel Santana and Tsang, Ren and Khademi Kalantari, Nima},
title = {Single Image HDR Reconstruction Using a CNN with Masked Features and Perceptual Loss},
journal = {ACM Transactions on Graphics},
volume = {39},
number = {4},
year = {2020},
month = {7},
doi = {10.1145/3386569.3392403}
}
```
