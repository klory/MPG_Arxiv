import os
import json
import numpy as np
import copy
import json
import argparse
from torchvision import transforms
from PIL import Image
import math
import cv2
import torch
from torch.nn import functional as F
import pathlib

ROOT = pathlib.Path(__file__).parent.resolve()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def normalize(img):
    """
    normalize a batch
    """
    img = (img-img.min())/(img.max()-img.min())
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:,i] = (img[:,i]-means[i])/stds[i]
    return img

def resize(img, size=224):
    """
    resize a batch
    """
    return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)


def load_categories(filename):
    with open(filename, 'r') as f:
        categories = f.read().strip().split('\n')
    return categories


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def infinite_loader(loader):
    """
    arguments:
        loader: torch.utils.data.DataLoader
    return:
        one batch of data
    usage:
        data = next(infinite_loader(loader))
    """
    while True:
        for batch in loader:
            yield batch

def make_captioned_image(caption, image, font=20, color=(255,255,0), loc=(0,0), nrow=8, tint_color=(0,0,0), opacity=0.4, pad_value=0):
    import torch
    from PIL import Image, ImageFont, ImageDraw
    from torchvision.utils import make_grid
    assert len(caption) == len(image)
    from sys import platform
    if platform == "linux" or platform == "linux2":
        font = ImageFont.truetype('UbuntuMono-R.ttf', font)
    elif platform == "darwin" or platform == 'win32':
        font = ImageFont.truetype('Arial.ttf', font)

    imgs = image.cpu().numpy()
    imgs = (imgs-imgs.min()) / (imgs.max()-imgs.min())
    txted_imgs = []
    # how to draw transparent rectangle: https://stackoverflow.com/a/43620169/6888630
    opacity = int(255*opacity)
    for txt, img in zip(caption, imgs):
        img = img.transpose(1,2,0)
        img = Image.fromarray(np.uint8(img*255))
        
        # draw background
        img = img.convert("RGBA")
        x=y=0
        w, h = font.getsize(txt)
        h *= (len(txt.split('\n'))+1)
        overlay = Image.new('RGBA', img.size, tint_color+(0,))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle(((x,y,x+w,y+h)), fill=tint_color+(opacity,))

        # Alpha composite these two images together to obtain the desired result.
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB") # Remove alpha for saving in jpg format.
        
        # draw text
        draw = ImageDraw.Draw(img)
        draw.text(loc, txt, fill=color, font=font)

        img = transforms.ToTensor()(img)
        txted_imgs.append(img)
    txted_imgs = torch.stack(txted_imgs)
    big_pic = make_grid(txted_imgs, nrow=nrow, padding=2, normalize=True, pad_value=pad_value, scale_each=True)
    return big_pic

def save_captioned_image(caption, image, fp, font=20, color=(255,255,0), opacity=0.5, loc=(0,0), nrow=8, pad_value=0):
    grid = make_captioned_image(caption, image, opacity=opacity, font=font, color=color, loc=loc, nrow=nrow, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)

def load_recipes(file_path, part='', food_type=''):
    print(f'load recipes from {file_path}')
    with open(file_path, 'r') as f:
        info = json.load(f)
    if part:
        info = [x for x in info if x['partition']==part]
    if food_type:
        info = [x for x in info if food_type in x['title'].lower()]
    return info


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)