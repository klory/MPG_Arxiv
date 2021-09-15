import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import lmdb
from io import BytesIO
from glob import glob

import sys
sys.path.append('../')
import common
from datasets import utils

def _load_one_pizza(idx, env, transform):
    with env.begin(write=False) as txn:
        key = f'{idx}'.encode('utf-8')
        ingrs = txn.get(key).decode('utf-8')
        if not ingrs:
            ingrs = 'empty'
        txt = ingrs

        key = f'{256}-{idx}'.encode('utf-8')
        img_bytes = txn.get(key)
        
    buffer = BytesIO(img_bytes)
    img = Image.open(buffer)
    img = transform(img)
    return img, txt

def _load_one_pizza_stackgan2(idx, env, transform, sizes=[64, 128, 256]):
    img, txt = _load_one_pizza(idx, env, transform)
    imgs = []
    for size in sorted(sizes)[::-1]:
        img = F.interpolate(img.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)
        imgs.append(img)
    return imgs[::-1], txt
 
class Pizza10DatasetFromImage(torch.utils.data.Dataset):
    def __init__(
        self,
        part='all',
        data_dir=f'{common.ROOT}/data/Pizza10/',
        transform=utils.resnet_transform_train,
    ):
        img_dir = f'{data_dir}/images'
        self.names = sorted(glob(f'{img_dir}/*.jpg'))
        self.transform = transform
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))

        # split by training and validation (8/2)
        split_point = int(0.8 * len(self.names))
        if part == 'train':
            self.names = self.names[:split_point]
            self.labels = self.labels[:split_point]
        elif part == 'val':
            self.names = self.names[split_point:]
            self.labels = self.labels[split_point:]
        
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        filename = self.names[index]
        label = self.labels[index]
        img = Image.open(filename).convert('RGB')
        img = self.transform(img)
        tgt = {}
        tgt['raw_label'] = utils.label2ingredients(label, self.categories)
        tgt['ingr_label'] = label
        return img, tgt


class Pizza10Dataset(Dataset):
    def __init__(
       self,
       data_dir=f'{common.ROOT}/data/Pizza10/',
       part='all', 
       transform=utils.resnet_transform_train
    ):
        lmdb_file = os.path.join(data_dir, 'data.lmdb')
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))
        length = len(self.labels)

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        self.names = range(length)
        # split by training and validation (8/2)
        split_point = int(0.8 * length)
        if part == 'train':
            self.names = self.names[:split_point]
        elif part == 'val':
            self.names = self.names[split_point:]

        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'images/')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # image
        idx = self.names[index]

        img, txt = _load_one_pizza(idx, self.env, self.transform)
        label = self.labels[idx]
        
        target = {}
        target['raw_label'] = txt
        target['ingr_label'] = label.float()
        return img, target


class Pizza10DatasetRetrieval(Dataset):
    def __init__(
        self,
        data_dir=f'{common.ROOT}/data/Pizza10/',
        part='train', 
        transform=utils.resnet_transform_train
    ):
        lmdb_file = os.path.join(data_dir, 'data.lmdb')
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))

        from collections import defaultdict
        label_dict= defaultdict(list)
        for idx, line in enumerate(self.labels):
            label_dict[tuple(line.tolist())].append(idx)
        
        items = label_dict.items()
        items = sorted(items, key=lambda x: -len(x[1]))
        filename = os.path.join(data_dir, 'statistics.json')
        with open(filename, 'w') as f:
            import json
            json.dump(items, f, indent=2)
        
        self.label_list = list(label_dict.values())
        self.class_sample_count = [len(x) for x in self.label_list]
        
        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        assert transform != None, 'transform can not be None!'
        self.transform = transform
        self.part = part

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        pizzas = self.label_list[idx]
        if self.part=="train":
            index = pizzas[int(3*np.random.random()/5*len(pizzas))]
        else:
            index = pizzas[int(3*len(pizzas)/5)+int(2*np.random.random()*len(pizzas)/5)]
        img, txt = _load_one_pizza(index, self.env, self.transform)
        return img, txt


class Pizza10DatasetMPG(Dataset):
    def __init__(
       self,
       data_dir=f'{common.ROOT}/data/Pizza10/',
       part='all', 
       transform=utils.gan_transform
    ):
        lmdb_file = os.path.join(data_dir, 'data.lmdb')
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))
        length = len(self.labels)

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        self.names = range(length)
        # split by training and validation (8/2)
        split_point = int(0.8 * length)
        if part == 'train':
            self.names = self.names[:split_point]
        elif part == 'val':
            self.names = self.names[split_point:]

        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'images/')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # image
        idx = self.names[index]
        img, txt = _load_one_pizza(idx, self.env, self.transform)
        label = self.labels[idx]
        
        target = {}
        target['raw_label'] = txt
        target['ingr_label'] = label.float()

        wrong_idx = np.random.choice(self.names)
        while self.labels[idx].tolist() == self.labels[wrong_idx].tolist():
            wrong_idx = np.random.choice(self.names)
        wrong_img, wrong_txt = _load_one_pizza(wrong_idx, self.env, self.transform)

        return img, target, wrong_img


class Pizza10DatasetMPG(Dataset):
    def __init__(
       self,
       data_dir=f'{common.ROOT}/data/Pizza10/',
       part='all', 
       transform=utils.gan_transform
    ):
        lmdb_file = os.path.join(data_dir, 'data.lmdb')
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))
        length = len(self.labels)

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        self.names = range(length)
        # split by training and validation (8/2)
        split_point = int(0.8 * length)
        if part == 'train':
            self.names = self.names[:split_point]
        elif part == 'val':
            self.names = self.names[split_point:]

        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'images/')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # image
        idx = self.names[index]
        img, txt = _load_one_pizza(idx, self.env, self.transform)
        label = self.labels[idx]
        
        target = {}
        target['raw_label'] = txt
        target['ingr_label'] = label.float()

        wrong_idx = np.random.choice(self.names)
        while self.labels[idx].tolist() == self.labels[wrong_idx].tolist():
            wrong_idx = np.random.choice(self.names)
        wrong_img, wrong_txt = _load_one_pizza(wrong_idx, self.env, self.transform)

        return img, target, wrong_img


class Pizza10DatasetStackGAN2(Dataset):
    def __init__(
       self,
       data_dir=f'{common.ROOT}/data/Pizza10/',
       part='all', 
       transform=utils.gan_transform
    ):
        lmdb_file = os.path.join(data_dir, 'data.lmdb')
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))
        length = len(self.labels)

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        self.names = range(length)
        # split by training and validation (8/2)
        split_point = int(0.8 * length)
        if part == 'train':
            self.names = self.names[:split_point]
        elif part == 'val':
            self.names = self.names[split_point:]

        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'images/')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # image
        idx = self.names[index]
        imgs, txt = _load_one_pizza_stackgan2(idx, self.env, self.transform)

        wrong_idx = np.random.choice(self.names)
        while self.labels[idx].tolist() == self.labels[wrong_idx].tolist():
            wrong_idx = np.random.choice(self.names)
        wrong_imgs, wrong_txt = _load_one_pizza_stackgan2(wrong_idx, self.env, self.transform)

        return txt, imgs, wrong_imgs


if __name__ == '__main__':
    from tqdm import tqdm
    bs = 64
    nrow = int(np.sqrt(64))
    os.makedirs('ignore/', exist_ok=True)
    torch.manual_seed(8)

    dataset = Pizza10Dataset(data_dir=f'{common.ROOT}/data/Pizza10/', part='train', transform=utils.resnet_transform_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    print(len(dataset), len(dataloader))
    for img, tgt in tqdm(dataloader):
        print(img.shape)
        # print(tgt['raw_label'])
        # print(tgt['ingr_label'])
        common.save_captioned_image(tgt['raw_label'], img, 'ignore/pizza10_batch.jpg', font=10, nrow=nrow)
        break

    dataset = Pizza10DatasetRetrieval(data_dir=f'{common.ROOT}/data/Pizza10/', part='train', transform=utils.resnet_transform_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    print(len(dataset), len(dataloader))
    for img, txt in tqdm(dataloader):
        print(img.shape)
        print(txt)
        break

    
    dataset = Pizza10DatasetMPG(data_dir=f'{common.ROOT}/data/Pizza10/', part='train', transform=utils.gan_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    print(len(dataset), len(dataloader))
    for img, tgt, wrong_img in tqdm(dataloader):
        print(img.shape)
        print(wrong_img.shape)
        # print(tgt['raw_label'])
        # print(tgt['ingr_label'])
        common.save_captioned_image(tgt['raw_label'], img, 'ignore/pizza10_batch.jpg', font=10, nrow=nrow)
        break