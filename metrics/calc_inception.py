import pickle

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception3
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
from metrics.inception import InceptionV3
import common

class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)
    feature_list = []
    
    for img, _ in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)
    return features


if __name__ == '__main__':
    from metrics.utils import load_args
    from datasets import utils
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = load_args()

    common.set_random_seed(args.seed)

    inception = load_patched_inception_v3()
    inception = nn.DataParallel(inception).eval().to(device)

    if 'pizza10' in args.dataset:
        from datasets.pizza10 import Pizza10Dataset
        dataset = Pizza10Dataset(transform=utils.resnet_transform_val)
    else:
        raise Exception('Unsupported dataset!')
    
    assert len(dataset) >= args.n_sample
    dataset = torch.utils.data.Subset(dataset, indices=np.random.choice(len(dataset), args.n_sample, replace=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    print(len(dataset), len(loader))

    features = extract_features(loader, inception, device).numpy()
    print(f'extracted {features.shape[0]} features')

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    with open(f'inception_{args.dataset}.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.size, 'dataset': args.dataset}, f)