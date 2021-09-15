import os
import numpy as np
from matplotlib import pyplot as plt
from time import time
import pdb
import argparse

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_ranks(rcps, imgs, retrieved_type='recipe'):
    assert imgs.shape == rcps.shape, 'recipe features and image features should have same dimension'
    # pdb.set_trace()
    imgs = imgs / np.linalg.norm(imgs, axis=1)[:, None]
    rcps = rcps / np.linalg.norm(rcps, axis=1)[:, None]
    if retrieved_type == 'recipe':
        sims = np.dot(imgs, rcps.T) # [N, N]
    else:
        sims = np.dot(rcps, imgs.T)
    
    ranks = []
    preds = []
    # loop through the N similarities for images
    for ii in range(imgs.shape[0]):
        # get a column of similarities for image ii
        sim = sims[ii,:]
        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()
        # find where the index of the pair sample ended up in the sorting
        pos = sorting.index(ii)
        ranks.append(pos+1.0)
        preds.append(sorting[0])
    # pdb.set_trace()
    return np.asarray(ranks), preds

def rank(rcps, imgs, retrieved_type='recipe', retrieved_range=1000, draw_hist=False, verbose=False, epoch=-1):
    t1 = time()
    N = retrieved_range
    data_size = imgs.shape[0]
    glob_rank = []
    glob_recall = {1:0.0, 5:0.0, 10:0.0}
    if draw_hist:
        plt.figure(figsize=(16, 6))
    # average over 10 sets
    for i in range(10):
        ids_sub = np.random.choice(data_size, N, replace=False)
        imgs_sub = imgs[ids_sub, :]
        rcps_sub = rcps[ids_sub, :]
        # loop through the N similarities for images
        ranks, _ = compute_ranks(rcps_sub, imgs_sub, retrieved_type)
        recall = {1:0.0, 5:0.0, 10:0.0}
        for ii in recall.keys():
            recall[ii] = (ranks<=ii).sum() / ranks.shape[0]
        med = int(np.median(ranks))
        for ii in recall.keys():
            glob_recall[ii] += recall[ii]
        glob_rank.append(med)
        if draw_hist:
            ranks = np.array(ranks)
            plt.subplot(2,5,i+1)
            n, bins, patches = plt.hist(x=ranks, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.ylim(top=300)
            # plt.xlabel('Rank')
            # plt.ylabel('Frequency')
            # plt.title('Rank Distribution')
            plt.text(23, 45, 'avgR(std) = {:.2f}({:.2f})\nmedR={:.2f}\n#<{:d}:{:d}|#={:d}:{:d}|#>{:d}:{:d}'.format(
                np.mean(ranks), np.std(ranks), np.median(ranks), 
                med,(ranks<med).sum(), med,(ranks==med).sum(), med,(ranks>med).sum()))
    if draw_hist:
        plt.savefig(f'hist_{epoch}.png')
    
    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10

    medR = np.mean(glob_rank)
    medR_std = np.std(glob_rank)
    t2 = time()
    if verbose:
        print(f'=>retrieved_range={retrieved_range}, MedR={medR:.4f}({medR_std:.4f}), time={t2-t1:.4f}s')
    return medR, medR_std, glob_recall