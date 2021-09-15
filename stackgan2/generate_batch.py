import torch
from torch._C import device
from torchvision import transforms
from PIL import Image
import os
import lmdb
from torch.utils import data
import json
from io import BytesIO
import numpy as np

import sys
sys.path.append('../')
from datasets.pizza10 import Pizza10Dataset
from datasets.utils import gan_transform
from common import ROOT, infinite_loader, requires_grad
from stackgan2.train import load_stackgan2_model
from retrieval_model.train import load_retrieval_model, compute_txt_feat

class BatchGenerator():
    def __init__(self, args):
        device = args.device
        
        ckpt_path = ROOT / args.ckpt_path
        ckpt_args, _, netG, _, _, _ = load_stackgan2_model(ckpt_path)
        netG = netG.eval().to(device)

        retrieval_model_path = ROOT / args.retrieval_model
        _, _, tokenizer, txt_encoder, _, _ = load_retrieval_model(retrieval_model_path, device)

        txt_encoder = txt_encoder.eval().to(device)
        requires_grad(txt_encoder, False)
        requires_grad(netG, False)
        
        if args.dataset_name == 'pizza10':
            dataset = Pizza10Dataset(transform=gan_transform)
        else:
            raise Exception('Unsupported dataset!')
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4)
        
        self.ckpt_args = ckpt_args
        self.netG = netG
        self.tokenizer = tokenizer
        self.txt_encoder = txt_encoder
        self.dataloader = infinite_loader(dataloader)
        self.batch_size = args.batch_size
        self.device = device
        self.fixed_noise = torch.randn(self.batch_size, self.ckpt_args.z_dim).to(self.device)

    def generate_ssim(self):
        _, real, fake, _ = self.generate_all()
        return real, fake
    
    def generate_fid(self):
        _, _, fake, _ = self.generate_all()
        return fake

    def generate_MedR(self):
        txt, _, fake, _ = self.generate_all()
        return txt, fake

    def generate_mAP(self):
        _, _, fake, label = self.generate_all()
        return fake, label

    def generate_all(self):
        real, tgt = next(self.dataloader)
        txt = tgt['raw_label']
        label = tgt['ingr_label']
        noise = torch.randn(self.batch_size, self.ckpt_args.z_dim).to(self.device)
        txt_feat = compute_txt_feat(txt, self.tokenizer, self.txt_encoder, device=self.device)
        fakes, _, _ = self.netG(noise, txt_feat)
        fake = fakes[-1]
        return tgt['raw_label'], real.to(self.device), fake, label
        # txt: [BS], array of strings
        # e.g. [['Arugula\nTomato\nPepperoni'], ...]
        
        # real: [BS, 3, size, size]
        
        # fake: [BS, 3, size, size]
        
        # label
        #   Recipe1M: [1, 1, 1 ...]
        #   Pizza10: [BS, 10]
        


if __name__ == '__main__':
    import pdb
    from types import SimpleNamespace
    args = SimpleNamespace(
        ckpt_path=f'{ROOT}/stackgan2/runs/1q7grcoo/batch0.ckpt',
        batch_size=32,
        size=256,
        device='cuda',
    )

    batch_generator = BatchGenerator(args)

    # # Recipe1M
    # batch_generator = BatchGenerator(
    #     f'{ROOT}/stackgan2/runs/3otuw71c/batch25000.ckpt',
    #     batch_size=32, size=256, device='cuda')
    
    txt, real, fake_img, binary_label = batch_generator.generate_all()
    print(txt)
    print(real.shape)
    print(fake_img.shape)
    print(binary_label.shape)