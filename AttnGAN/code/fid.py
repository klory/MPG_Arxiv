import argparse
import pickle
from torch.utils import data
from torchvision import transforms, utils
import torch
from torch import nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from scipy import linalg
from tqdm import tqdm
import os
from datasets import Pizza10Dataset
from calc_inception import load_patched_inception_v3
from miscc.config import cfg
from miscc.config import cfg_from_file
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
import dict_util

def build_models(generator_path):
        cfg.TRAIN.NET_G=generator_path
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(15, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator############## #
        if cfg.GAN.B_DCGAN:
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
        else:
            netG = G_NET()

        netG.apply(weights_init)

        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
  
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
        return [text_encoder, image_encoder, netG, epoch]

@torch.no_grad()
def extract_all_from_samples(
    text_encoder, image_encoder,loader,netG, inception, truncation, truncation_latent, batch_size, n_sample
):
    
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid>0:
        batch_sizes = [batch_size] * n_batch + [resid]
    else:
        batch_sizes = [batch_size] * n_batch
    features = []
    path_length_list = []
    # requires_grad(generator, True)
    for batch in tqdm(batch_sizes):
        data = next(loader)
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        print(captions)
        print(cap_lens)
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        #######################################################
        # (2) Generate fake images
        ######################################################
        nz = cfg.GAN.Z_DIM
        # noise = Variable(torch.FloatTensor(args.batch, nz)).cuda()
        # noise.data.normal_(0, 1)
        # noise = torch.randn(args.batch, nz, device="cuda")
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True).cuda()
        img, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)


        # if args.conditional:
        #     c_code = textencoder.get_c_code(txt)
        #     sample_z = torch.cat((torch.randn(args.batch, 256, device=device),c_code),1)
        # else:
        #     sample_z = torch.randn(batch, 512, device="cuda")
        
        # img, _ = generator([sample_z], truncation=truncation, truncation_latent=truncation_latent)
        

        utils.save_image(
                            img[2],
                            f"p.png",
                            nrow=int(8),
                            normalize=True,
                            range=(-1, 1),
                            save=True
                        )

        feat = inception(img[2])[0].view(img[2].shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features.numpy(),0



def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
def infinite_loader(loader):
    while True:
        for image,captions,cap_lens,class_ids,keys in loader:
            yield image,captions,cap_lens,class_ids,keys

def extract_feature_from_reals(
    loader, inception, truncation, truncation_latent, batch_size, n_sample
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid>0:
        batch_sizes = [batch_size] * n_batch + [resid]
    else:
        batch_sizes = [batch_size] * n_batch
    features = []

    for batch in tqdm(batch_sizes):
        img, _, _, _, _ = next(loader)
        img = img[-1].to("cuda")
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))
    utils.save_image(
                        img,
                        f"real.png",
                        nrow=int(8),
                        normalize=True,
                        range=(-1, 1),
                        save=True
                    )
    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=100)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--food_type', type=str)
    parser.add_argument('--folder', type=str, default='../output/food_attn2_2020_10_28_18_39_59/Model/')
    
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/food_attn2.yml', type=str)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--conditional", action="store_true", default=False)
    parser.add_argument("--embedding_space", type=str)
    parser.add_argument("--file_name", type=str, default="salad_fid.txt")
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()
    cfg_from_file(args.cfg_file)
    dir_list = os.listdir(args.folder)
    dir_list.reverse()
    print("loading inception...")
    inception = load_patched_inception_v3().to("cuda")
    inception.eval()


    transform = transforms.Compose(
        [   transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    attn_transform = transforms.Compose([
        transforms.Scale(int(299 * 76 / 64)),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip()])

    dataset = Pizza10Dataset(part=None,base_size=cfg.TREE.BASE_SIZE,
                  transform=attn_transform,cfg=cfg)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )
    loader = infinite_loader(loader)
    real_features = extract_feature_from_reals(
        loader, inception, args.truncation,None, args.batch, args.n_sample
    ).numpy()
    real_mean = np.mean(real_features, 0)
    real_cov = np.cov(real_features, rowvar=False)
    score_dict = {}
    
    count = 0

    for file_name in [ 'netG_epoch_12000.pth', 'netG_epoch_11500.pth', 'netG_epoch_11000.pth', 'netG_epoch_10500.pth', 'netG_epoch_10000.pth', 'netG_epoch_9500.pth', 'netG_epoch_9000.pth', 'netG_epoch_8500.pth', 'netG_epoch_8000.pth', 'netG_epoch_7500.pth', 'netG_epoch_7000.pth', 'netG_epoch_6500.pth', 'netG_epoch_6000.pth', 'netG_epoch_5500.pth', 'netG_epoch_5000.pth', 'netG_epoch_4500.pth', 'netG_epoch_4000.pth', 'netG_epoch_3500.pth', 'netG_epoch_3000.pth', 'netG_epoch_2500.pth']:
        count = count+1
        if count%args.skip!=0:
            continue
        text_encoder, image_encoder, g, start_epoch = build_models(args.folder+file_name)
        
        g.eval()
        # if args.truncation < 1:
        #     with torch.no_grad():
        #         mean_latent = g.mean_latent(args.truncation_mean)

        # else:
        #     mean_latent = None
        print("loading checkpoint",file_name)
        print("calculating statistic")
        features,path_length_val = extract_all_from_samples(
            text_encoder, image_encoder,loader,g,inception, args.truncation, None, args.batch, args.n_sample
        )
        print(f'extracted {features.shape[0]} features')

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)
        print("calculating fid...")
        fid= calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        print(fid)
        tempdict =  {"ckpt":file_name[11:-5],'fid':fid}
        dict_util.save_dict(tempdict,args.file_name)
        score_dict[file_name[11:-5]] = {'fid':fid,"ppl":path_length_val}
    
    print(score_dict)
    


