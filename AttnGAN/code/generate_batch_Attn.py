from PIL.Image import fromarray
import torch
from torch import tensor
from torch._C import device
from torchvision import transforms, utils
from PIL import Image
import os
import lmdb
from torch.utils import data
import json
from io import BytesIO
import numpy as np
import sys
sys.path.append('../../')
from common import ROOT
from torch.autograd import Variable
from AttnGAN.code.miscc.config import cfg
from AttnGAN.code.miscc.config import cfg_from_file
from AttnGAN.code.model import G_DCGAN, G_NET
from AttnGAN.code.model import RNN_ENCODER, CNN_ENCODER
from AttnGAN.code.miscc.utils import weights_init, load_params, copy_G_params
from AttnGAN.code.datasets import get_ingredients_wordvec
from AttnGAN.code.miscc.losses import words_loss
from AttnGAN.code.miscc.utils import build_super_images, build_super_images2

sys.path.append('../retrieval_model')
# from train_ret import load_retrieval_model, compute_txt_feat
cfg_file = f'{ROOT}/AttnGAN/code/cfg/food_attn2.yml'
cfg_from_file(cfg_file)
# try to keep the transform intact
def prepare_data(data):
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

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])
def infinite_loader(loader):
    while True:
        for data in loader:
            yield data
def build_models(generator_path,encoder_path):

        cfg.TRAIN.NET_G=generator_path
        cfg.TRAIN.NET_E = encoder_path
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(12, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E,map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator############## #
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        else:
            netG = G_NET()

        netG.apply(weights_init)

        epoch = 0
        if cfg.TRAIN.NET_G != '':
            print(generator_path)
            state_dict = torch.load(generator_path, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
        return [text_encoder, image_encoder, netG, epoch]


class Pizza10Dataset(data.Dataset):
    def __init__(
        self, 
        lmdb_file=f'{ROOT}/data/pizzaGANdata_new_concise/pizzaGANdata.lmdb', 
        part='train', transform=None, return_image=True):
        base_size = cfg.TREE.BASE_SIZE
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        with open(f"{ROOT}/data/pizzaGANdata_new_concise/categories.txt", 'r') as f:
            self.categories = np.asarray(f.read().strip().split('\n'))
        dirname = os.path.dirname(lmdb_file)
        label_file = os.path.join(dirname, 'imageLabels.txt')
        with open(label_file, 'r') as f:
            self.labels = f.read().strip().split('\n')
        
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
        self.norm= transform

        self.part = part
        self.ixtoword, self.wordtoix,self.n_words=self.build_dictionary()
        self.return_image = return_image

    def __len__(self):
        return self.length
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

class BatchGenerator():
    def __init__(self, args):
        self.args = args
        assert args.dataset_name in ['pizzaGANdata_new_concise', 'Recipe1M']
        cfg_file = f'{ROOT}/AttnGAN/code/cfg/food_attn2.yml'
        cfg_from_file(cfg_file)

        device = args.device
        self.text_encoder, self.image_encoder, self.netG, _ = build_models(args.ckpt_path,cfg.TRAIN.NET_E)
        # self.netG.eval()
        dataset = None
        if 'pizzaGANdata_new_concise' == args.dataset_name:
            dataset = Pizza10Dataset(transform = transform)
        elif 'Recipe1M' == args.dataset_name:
            raise Exception('Unsupported dataset!')

        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True
        )

        self.original_dataloader = dataloader

        self.dataloader = infinite_loader(dataloader)
        self.batch_size = args.batch_size
        self.device = device
        self.dataset = dataset
        self.fixed_noise = torch.randn(self.args.batch_size, cfg.GAN.Z_DIM).to(self.device)

    def generate_ssim(self):
        _, real, batch_fake_img, _ = self.generate_all()
        return real, batch_fake_img

    def generate_fid(self):
        _, _, batch_fake_img, _ = self.generate_all()
        return batch_fake_img

    def generate_MedR(self):
        batch_txt, _, batch_fake_img, _ = self.generate_all()
        return batch_txt, batch_fake_img

    def generate_mAP(self):
        _, _, batch_fake_img, batch_binary_label = self.generate_all()
        return batch_fake_img, batch_binary_label
    
    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations,name='current',dataset = None):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, dataset.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (".", name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, dataset.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (".", name, gen_iterations)
            im.save(fullpath)
    @torch.no_grad()
    def generate_all(self):
        
        data = prepare_data(next(self.dataloader))
        imgs, captions, cap_lens, ingredients,binary_label = data
        # imgs, captions, cap_lens, ingredients,binary_label = prepare_data(data)

        hidden = self.text_encoder.init_hidden(self.batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        
        # print(ingredients)
        # captions = captions[0].repeat(32,1)
        # cap_lens = cap_lens[0].repeat(32)

        words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        #######################################################
        # (2) Generate fake images
        ######################################################
        noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM)).cuda()
        noise.data.normal_(0, 1)
        # print(captions.size())
        # print(cap_lens.size())
        # print(noise.size())
        # print(mask)
        # noise = torch.randn(self.batch_size, cfg.GAN.Z_DIM).cuda()
        batch_fake_img, _, mu, logvar = self.netG(noise, sent_emb, words_embs, mask)
        # self.save_img_results( self.netG, noise, sent_emb, words_embs, mask,
        #                  self.image_encoder, captions, cap_lens,
        #                  1000, "average",dataset=self.dataset)
        return ingredients, imgs[2].to(self.device), batch_fake_img[2], binary_label
        # batch_txt: [BS], array of strings
        # e.g. [['Arugula\nTomato\nPepperoni'], ...]
        
        # batch_fake_img: [BS, 3, size, size]
        
        # batch_binary_label
        #   Recipe1M: [1, 1, 1 ...]
        #   PizzaGANdata_new_concise: [BS, 10]
        





if __name__ == '__main__':
    import pdb

    from types import SimpleNamespace
    args = SimpleNamespace(
        ckpt_path=f'{ROOT}/AttnGAN/output/food_attn2_2020_11_10_11_51_01/Model/netG/netG_epoch_6000.pth',
        # ckpt_path=f'{ROOT}/AttnGAN/output/food_attn2_2020_10_28_18_39_59/Model/netG_epoch_12000.pth',
         
        cfg_file = f'{ROOT}/AttnGAN/code/cfg/food_attn2.yml',
        dataset_name='pizzaGANdata_new_concise',
        batch_size=32,
        size=256,
        device='cuda',
    )
    if args.cfg_file is not None:
        print("cfg from file ......")
        cfg_from_file(args.cfg_file)
    # pizzaGANdata_new_concise
    batch_generator = BatchGenerator(args)
    txt, fake_img, binary_label = batch_generator.generate_all()

    # for i in txt:
    #     print(i.split\n"))
    # print(binary_label)
    # utils.save_image(
    #                         fake_img,
    #                         f"p.png",
    #                         nrow=int(8),
    #                         normalize=True,
    #                         range=(-1, 1),
    #                         save=True
    #                     )
    assert len(txt) == 32
    print(txt)
    assert type(txt[0]) == str
    assert fake_img.shape == (32,3,256,256)
    assert fake_img.dtype == torch.float
    assert binary_label.size() == (32, 10)
    assert binary_label.dtype == torch.float

    # # TODO: test Recipe1M (similar to the above)
    # batch_generator = BatchGenerator(
    #     f'{ROOT}/AttnGAN/.../xxx.ckpt',
    #     batch_size=32, size=256, device='cuda')