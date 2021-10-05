import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchnet import meter

import pdb
import os
import csv
from glob import glob
from torch.nn import functional as F
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from common import requires_grad, normalize, resize, ROOT
from ingr_classifier.train import load_classifier


def compute_mAP(args, ingr_classifier):
    print(f'\nworking on {args.ckpt_path}')
    batch_generator = BatchGenerator(args)

    running_output = []
    running_label = []
    with torch.no_grad():
        for _ in tqdm(range(1000//args.batch_size)):
            fake_img, binary_label = batch_generator.generate_mAP()
            fake_img = normalize(resize(fake_img, size=224))
            output = ingr_classifier(fake_img)
            running_output.append(output)
            running_label.append(binary_label)
    
    running_output = torch.cat(running_output, dim=0)
    running_label = torch.cat(running_label, dim=0)
    
    mtr = meter.APMeter()
    mtr.add(running_output, running_label)
    APs = mtr.value()
    mAP = APs.mean().item() # mean average precision
    return mAP

if __name__ == '__main__':
    from metrics.utils import load_args 
    args = load_args()

    # assertations
    assert 'dataset' in args.__dict__
    assert 'ckpt_dir' in args.__dict__
    assert 'classifier' in args.__dict__
    assert 'device' in args.__dict__
    assert 'batch_size' in args.__dict__


    if 'stackgan2/' in args.ckpt_dir:
        from stackgan2.generate_batch import BatchGenerator
    elif 'AttnGAN/' in args.ckpt_dir:
        from AttnGAN.code.generate_batch_Attn import BatchGenerator
    elif 'mpg/' in args.ckpt_dir:
        from mpg.generate_batch import BatchGenerator
    
    args.ckpt_dir = ROOT / args.ckpt_dir
    args.classifier = ROOT / args.classifier
    device = args.device

    _, _, classifier, _ = load_classifier(args.classifier)
    classifier = classifier.eval().to(device)
    requires_grad(classifier, False)

    # *******************************
    # only run for one checkpoint
    # ********************************
    if not args.sweep:
        args.ckpt_path = ROOT / args.ckpt_path
        mAP = compute_mAP(args, classifier)
        print(f'mAP={mAP:.4f}')
        sys.exit(0)

        
    # *******************************
    # run for all checkpoints
    # ********************************
    filename = os.path.join(args.ckpt_dir, 'mAP.csv')
    # load values that are already computed
    computed_iterations = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                computed_iterations += [row[0]]
    
    # prepare to write
    f = open(filename, mode='a')
    writer = csv.writer(f, delimiter=',')

    # find checkpoints
    ckpt_paths = glob(os.path.join(args.ckpt_dir, '*.ckpt')) + glob(os.path.join(args.ckpt_dir, '*.pt'))+glob(os.path.join(args.ckpt_dir, '*.pth'))
    iterations = [os.path.basename(ckpt_path).split('.')[0] for ckpt_path in ckpt_paths]
    ckpt_paths = sorted(ckpt_paths)
    print('records:', iterations)
    print('computed_iterations:', computed_iterations)
    for ckpt_path in ckpt_paths:
        iteration = os.path.basename(ckpt_path).split('.')[0]
        if iteration in computed_iterations:
            print('already computed')
            continue
        
        args.ckpt_path = ckpt_path
        mAP = compute_mAP(args, classifier)
        print(f'{iteration}, mAP={mAP:.4f}')
        writer.writerow([iteration, mAP])

    
    f.close()
    mAPs = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            mAP = float(row[1])
            mAPs += [mAP]
    fig = plt.figure(figsize=(6,6))
    plt.plot(mAPs)
    plt.savefig(os.path.join(args.ckpt_dir, 'mAP.png'))