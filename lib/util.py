import cv2
import numpy as np
import os

from datetime import datetime
from pathlib import Path
from random import randint, shuffle

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

    def __str__(self):
        return "{} - {}".format(self.total, self.steps)

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_data_directory(im_dir):
    """ Return the training input and label directory for a given input directory 
        or the input image and its label for a given image. """
    def get_gt_path(name_jpg):
        file = os.path.splitext(name_jpg.split('/')[-1])[0] + '.bin'
        path = os.path.join(str(Path(os.path.dirname(name_jpg)).parent), 'bin')
        label_image = os.path.join(path, file)

        return label_image

    if os.path.isdir(im_dir):
        im_dir, label_dir = os.path.join(im_dir, 'jpg'), os.path.join(im_dir, 'bin')
    else:
        im_dir, label_dir = im_dir, get_gt_path(im_dir)

    return im_dir, label_dir

def current_timestamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def random_mask(height, width, channels=3):
    """Generates a random irregular mask with lines, circles and elipses"""    
    img = np.zeros((height, width, channels), np.float32)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")
    
    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
    return 1-img

def saturated_(im, th):
    return np.minimum(np.maximum(0.0, im.max(2) - th) / (1 - th), 1)

def saturated_channel_(im, th):
    return np.minimum(np.maximum(0.0, im - th) / (1 - th), 1)

def get_saturated_regions(im, th=0.95):
    w,h,ch = im.shape

    mask_conv = np.zeros_like(im)
    for i in range(ch):
        mask_conv[:,:,i] = saturated_channel_(im[:,:,i], th)

    return mask_conv#, mask

def get_binary_saturated_regions(im, th=0.90):
    """ 
        Obtain the saturated regions. A linear ramp starting 
        from pixel values at a threshold (th) and ending at 
        the maximum pixel value.
        
        Keyword arguments:
        --------------------------------------------------
        im -- input image in the range [0,1]
        th -- threshold (default 0.95)
    """
    mask = saturated_(im, th)
    _, hard_mask = cv2.threshold(mask, 0.1, 1.0, cv2.THRESH_BINARY)

    hard_mask = np.dstack([hard_mask, hard_mask, hard_mask])

    return hard_mask