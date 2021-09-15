# CUDA_VISIBLE_DEVICES=0,1 python simple_generate.py --ckpt=runs/1acrnt5t/010000.pt
import argparse

import torch
import torchvision
import os
from tqdm import tqdm
import math
from torch.utils.data import Dataset
import numpy as np
import lmdb
from torch.utils import data
from PIL import Image
from io import BytesIO
from torchvision import transforms
import sys
from torch.autograd import Variable
sys.path.append('../../')
from AttnGAN.code.generate_batch_Attn import build_models
from AttnGAN.code.miscc.config import cfg
from AttnGAN.code.miscc.config import cfg_from_file
from AttnGAN.code.datasets import get_ingredients_wordvec
from common import ROOT

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])
def prepare_data2(data):
    imgs, captions, captions_lens,ingredients,binary_label = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    binary_label = binary_label[sorted_cap_indices]
    ingredients = np.array(ingredients)[sorted_cap_indices.numpy()]
    # binary_label = np.array(binary_label)[sorted_cap_indices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        
    return [real_imgs, captions, sorted_cap_lens,
            ingredients.tolist(),binary_label]
def prepare_data(data):
    imgs, captions, captions_lens,ingredients,binary_label = data

    real_imgs = []

    for i in range(0,3):

        real_imgs.append(Variable(imgs[i]).cuda())
    new_captions = Variable(torch.from_numpy(captions)).squeeze().unsqueeze(0).cuda().repeat(8,1)
    new_cap_lens =  Variable(torch.LongTensor([captions_lens])).cuda().repeat(8)

    return [real_imgs,new_captions,new_cap_lens,
            ingredients,binary_label]

class Pizza10Dataset(Dataset):
    def __init__(
        self, 
        lmdb_file=f'{ROOT}/data/pizzaGANdata_new_concise/pizzaGANdata.lmdb', 
        part='train', transform=None, return_image=True):
        base_size = cfg.TREE.BASE_SIZE
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        dirname = os.path.dirname(lmdb_file)
        label_file = os.path.join(dirname, 'imageLabels.txt')
        with open(label_file, 'r') as f:
            self.labels = f.read().strip().split('\n')
        with open(f"{ROOT}/data/pizzaGANdata_new_concise/categories.txt", 'r') as f:
            self.categories = np.asarray(f.read().strip().split('\n'))
        label_dict= {}
        for idx,line in enumerate(self.labels):
            if not line in label_dict:
                label_dict[line] = []
            label_dict[line].append(idx)
        
        items = label_dict.items()
        self.items = sorted(items, key=lambda x: -len(x[1]))
        self.ixtoword, self.wordtoix,self.n_words=self.build_dictionary()
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

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))



        assert transform!=None, 'transform can not be None!'
        self.transform = transform

        self.part = part

        self.return_image = return_image
    def build_dictionary(self):
        vocab = self.categories.tolist()
        vocab.append("<other>")
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        return ixtoword,wordtoix, len(ixtoword)
    def __len__(self):
        return len(self.items)
    def get_caption(self, sent_caption):
        
        sent_caption=sent_caption[sent_caption != 0]
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len
    def _load_pizza(self, idx):
        imgs = []
        with self.env.begin(write=False) as txn:

            for i in self.imsize:
                key = f'{i}-{idx}'.encode('utf-8')
                img_bytes = txn.get(key)
                buffer = BytesIO(img_bytes)
                img = Image.open(buffer)
                img = self.transform(img)
                imgs.append(img)
            
            label = [int(x) for x in self.labels[idx].split()]
            ingredients = [self.categories[i] for i in np.asarray(label).nonzero()[0]]
            if not ingredients:
                ingredients = ["<other>"]
            ingredients,_=get_ingredients_wordvec(ingredients,self.wordtoix)
            sent_caption = np.asarray(ingredients).astype('int64')
            txt, x_len = self.get_caption(sent_caption)

            key = f'{idx}'.encode('utf-8')
            ingredients = txn.get(key).decode('utf-8')
            
        return imgs,txt,x_len,ingredients

    def __getitem__(self, idx):
        binary_label = torch.FloatTensor([int(x) for x in self.labels[idx].split()])
        imgs,txt,x_len,ingredients = self._load_pizza(idx)
        return imgs,txt,x_len,ingredients,binary_label
def infinite_loader(loader):
    while True:
        for data in loader:
            yield data

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
@torch.no_grad()
def generate_cond(args, text_encoder, netG, dataset):
    netG.eval()
    save_dir = os.path.join(os.path.dirname(args.ckpt_path), 'outputs')
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, 'captions.txt'), 'w')
    dataloader = data.DataLoader(
            dataset,
            batch_size=8,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True
        )
    dataloader = infinite_loader(dataloader)
    with torch.no_grad():
        for i in tqdm(range(args.pics)):
            img, captions, cap_lens, ingredients,binary_label = prepare_data(dataset[i])
            # img, captions, cap_lens, ingredients,binary_label = prepare_data2(next(dataloader))
            f.write(str(i)+'\n')
            f.write(ingredients)
            f.write('\n')
            hidden = text_encoder.init_hidden(args.sample)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            # print(words_embs.size())
            # print(sent_emb.size())
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            #######################################################
            # (2) Generate fake images
            ######################################################
            noise = Variable(torch.FloatTensor(args.sample, cfg.GAN.Z_DIM)).cuda()
            noise.data.normal_(0, 1)
            # noise = torch.randn(self.batch_size, cfg.GAN.Z_DIM).cuda()
            sample, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)   
            sample = sample[2]
            sample = torch.cat([img[2].unsqueeze(0).cpu(), sample.detach().cpu()], dim=0)
            name = f'trunc={args.truncation:.2f}_'
            name += '|'.join(ingredients.split("\n"))
            torchvision.utils.save_image(
                sample,
                os.path.join(save_dir, f"{name}.png"),
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )
    f.close()

@torch.no_grad()
def generate_one(noise, binary_label, text_encoder, netG):
    netG.eval()
    dataset = Pizza10Dataset(transform=transform, return_image=True)
    i2w = dataset.ixtoword
    binary_label = binary_label.cpu().numpy()
    captions = []
    print(binary_label)
    for i, num in enumerate(binary_label[0]):
        if num!=0:
            captions.append(i+1)
    print(captions)
    if len(captions)==0:
        captions.append(11)
    captions = np.asarray(captions).astype('int64')
    # captions = ['\n'.join([i2w[idx] for idx in binary_label.nonzero()[0]])]
    captions, cap_lens = dataset.get_caption(captions)
    captions = torch.from_numpy(captions).squeeze(1).unsqueeze(0).repeat(2,1).cuda()
    cap_lens = torch.Tensor([cap_lens]).repeat(2).cuda()
    noise = noise.repeat(2,1).cuda()
    print(captions.size())
    print(cap_lens.size())
    print(noise.size())
    hidden = text_encoder.init_hidden(2)
    # words_embs: batch_size x nef x seq_len
    # sent_emb: batch_size x nef
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    # print(words_embs.size())
    # print(sent_emb.size())
    mask = (captions == 0)
    num_words = words_embs.size(2)
    if mask.size(1) > num_words:
        mask = mask[:, :num_words]
    print(mask)
    #######################################################
    # (2) Generate fake images
    ######################################################
    sample, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)   
    sample = sample[2]
    return sample[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=f"{ROOT}/AttnGAN/output/food_attn2_2020_11_10_11_51_01/Model/netG/netG_epoch_6000.pth",
        help="path to the model checkpoint",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=8,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )

    args = parser.parse_args()

    device = "cuda"

    
    cfg_from_file(f'{ROOT}/AttnGAN/code/cfg/food_attn2.yml')
    text_encoder, image_encoder, netG, _ = build_models(args.ckpt_path)
    text_encoder.eval()
    image_encoder.eval()

    # # ckpt_args, _, label_encoder, _, _, g_ema, _, _, _ = load_mpg(args.ckpt_path, device=device)

    # if args.truncation < 1:
    #     with torch.no_grad():
    #         mean_latent = g_ema.mean_latent(args.truncation_mean)
    # else:
    #     mean_latent = None
    
    # label_encoder, g_ema = [torch.nn.DataParallel(x.eval().to(device)) for x in [label_encoder, g_ema]]


    dataset = Pizza10Dataset(
        transform=transform, return_image=True)

    print('generate images')
    generate_cond(args, text_encoder, netG, dataset, device)
