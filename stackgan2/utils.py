import torch
from torch.nn import functional as F
import torchvision.utils as vutils
import pdb

def prepare_data(data, device):
    txt, imgs, w_imgs = data
    real_vimgs, wrong_vimgs = [], []
    for i in range(len(imgs)):
        real_vimgs.append(imgs[i].to(device))
        wrong_vimgs.append(w_imgs[i].to(device))
    return txt, real_vimgs, wrong_vimgs


def renormalize(img):
    '''
    img: [-1,1]
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img/2 + 0.5
    img = F.interpolate(img, [224, 224], mode='bilinear', align_corners=True)
    for i in range(img.shape[1]):
        img[:,i] = (img[:,i]-mean[i])/std[i]
    return img

def save_img_results(real_imgs, fake_imgs, save_dir, epoch, level=-1):
    num = 64
    real_img = real_imgs[level][0:num]
    fake_img = fake_imgs[level][0:num]
    real_fake = torch.stack([real_img, fake_img]).permute(1,0,2,3,4).contiguous()
    real_fake = real_fake.view(-1, real_fake.shape[-3], real_fake.shape[-2], real_fake.shape[-1])
    vutils.save_image(
            real_fake, 
            '{}/e{}_real_fake.png'.format(save_dir, epoch),  
            normalize=True, scale_each=True)
    real_fake = vutils.make_grid(real_fake, normalize=True, scale_each=True)
    vutils.save_image(
            fake_img, 
            '{}/e{}_fake_samples.png'.format(save_dir, epoch), 
            normalize=True, scale_each=True)
    return real_fake