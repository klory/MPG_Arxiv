import torch
from tqdm import tqdm
import numpy as np
import argparse
import os

import sys
sys.path.append('../')
from common import ROOT
import retrieval_model.utils as utils
from retrieval_model.train import load_retrieval_model, compute_txt_feat, compute_img_feat
from datasets.pizza10 import Pizza10DatasetRetrieval
from datasets.utils import resnet_transform_train, resnet_transform_val

# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

@torch.no_grad()
def validate(args, val_loader, tokenizer, txt_encoder, img_encoder, device):
    print('val')
    txt_encoder.eval()
    img_encoder.eval()

    txt_outputs = []
    img_outputs = []
    for img, txt in tqdm(val_loader):
        txt_output = compute_txt_feat(txt, tokenizer, txt_encoder, device=device)
        img_output = compute_img_feat(img, img_encoder, device=device)

        txt_outputs.append(txt_output.detach().cpu())
        img_outputs.append(img_output.detach().cpu())

    txt_outputs = torch.cat(txt_outputs, dim=0).numpy()
    img_outputs = torch.cat(img_outputs, dim=0).numpy()

    retrieved_range = min(txt_outputs.shape[0], args.retrieved_range)
    medR, medR_std, recalls = utils.rank(
        txt_outputs, img_outputs, retrieved_type=args.retrieved_type, 
        retrieved_range=retrieved_range, verbose=True)


if __name__ == '__main__':
    from retrieval_model.args import get_args
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = args.device

    ckpt_args, _, tokenizer, txt_encoder, img_encoder, _ = load_retrieval_model(args.ckpt_path, device)
    print(ckpt_args)

    val_set = Pizza10DatasetRetrieval(data_dir=f'{ROOT}/data/Pizza10/', part='val', transform=resnet_transform_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=2*ckpt_args.batch_size, num_workers=ckpt_args.workers)
    print(len(val_set), len(val_loader))

    validate(args, val_loader, tokenizer, txt_encoder, img_encoder, device)