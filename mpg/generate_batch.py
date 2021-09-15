import torch
from torchvision import transforms
from PIL import Image
import os
import lmdb
from torch.utils import data
import json
from io import BytesIO
import numpy as np
import pdb

import sys
sys.path.append('../')
from common import infinite_loader, ROOT, requires_grad
from mpg.train import load_mpg
from datasets.pizza10 import Pizza10Dataset
from datasets.utils import gan_transform


class BatchGenerator():
    def __init__(self, args):
        device = args.device
        ckpt_path = ROOT / args.ckpt_path
        ckpt_args, _, label_encoder, _, _, netG, _, _, _ = load_mpg(ckpt_path, device=device)
        label_encoder = label_encoder.eval().to(device)
        netG = netG.eval().to(device)
        requires_grad(label_encoder, False)
        requires_grad(netG, False)
        if args.dataset_name == 'pizza10':
            dataset = Pizza10Dataset(transform=gan_transform)
        else:
            raise Exception('Unsupported dataset!')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
        self.dataloader = infinite_loader(dataloader)
        self.ckpt_args = ckpt_args
        self.label_encoder = label_encoder
        self.netG = netG
        self.batch_size = args.batch_size
        self.truncation = args.truncation
        self.device = device
        self.fixed_noise = [torch.randn(args.batch_size, ckpt_args.style_dim, device=device)]
        if args.truncation < 1:
            self.mean_latent = netG.mean_latent(args.truncation_mean)
        else:
            self.mean_latent = None

    def generate_ssim(self):
        _, img, fake_img, _ = self.generate_all()
        return img, fake_img

    def generate_fid(self):
        _, _, fake_img, _ = self.generate_all()
        return fake_img

    def generate_MedR(self):
        batch_txt, _, fake_img, _ = self.generate_all()
        return batch_txt, fake_img

    def generate_mAP(self):
        _, _, fake_img, label = self.generate_all()
        return fake_img, label

    def generate_all(self, noise_is_same=False):
        img, tgt = next(self.dataloader)
        label = tgt['ingr_label']
        img = img.to(self.device)
        label = label.to(self.device)
        if noise_is_same:
            noise = self.fixed_noise
        else:
            noise = [torch.randn_like(self.fixed_noise[0])]
        txt_feat = self.label_encoder(label)
        fake_img, _ = self.netG(
            noise, txt_feat, input_is_latent=self.ckpt_args.input_is_latent, 
            truncation=self.truncation, truncation_latent=self.mean_latent)
        return tgt['raw_label'], img, fake_img, label
        # batch_txt: [BS], array of strings
        # e.g. [['Arugula\nTomato\nPepperoni'], ...]
        # fake_img: [BS, 3, size, size]
        # label
        #   Recipe1M: [1, 1, 1 ...]
        #   PizzaGAN10: [BS, 10]
        


if __name__ == '__main__':
    import pdb

    from types import SimpleNamespace
    args = SimpleNamespace(
        ckpt_path=f'{ROOT}/mpg/runs/2y8walj3/010000.pt',
        batch_size=32,
        size=256,
        device='cuda',
        truncation=0.5,
        truncation_mean=4096
    )

    # pizza10
    batch_generator = BatchGenerator(args)
    txt, img, fake_img, binary_label = batch_generator.generate_all()
    print(txt)
    print(img.shape)
    print(fake_img.shape)
    print(binary_label.shape)