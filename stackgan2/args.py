import argparse
import sys
sys.path.append('../')
from common import ROOT

def get_parser():
    parser = argparse.ArgumentParser(description='Train a GAN network')

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--wandb', type=int, default=1, choices=[0,1])

    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_batches', type=int, default=200_000)

    parser.add_argument('--base_size', type=int, default=64)
    parser.add_argument('--levels', type=int, default=3)

    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=100)

    parser.add_argument('--labels', type=str, default='original', choices=['original', 'R-smooth', 'R-flip', 'R-flip-smooth'])
    parser.add_argument("--input_noise", type=int, default=0)
    parser.add_argument('--uncond', type=float, default=1.0)
    parser.add_argument('--cycle_txt', type=float, default=0.0)
    parser.add_argument('--cycle_img', type=float, default=1.0)
    # parser.add_argument('--tri_loss', type=float, default=0.0)
    parser.add_argument('--kl', type=float, default=2.0)

    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)

    parser.add_argument('--ckpt_path', type=str, default='')

    # parser.add_argument('--lmdb_file', type=str, default=f'{ROOT}/data/Recipe1M/Recipe1M.lmdb')
    # parser.add_argument('--retrieval_model', type=str, default=f'{ROOT}/retrieval_model/runs/1p6y5lhs/e79.pt')

    parser.add_argument('--dataset', type=str, default='pizza10', choices=['pizza10'])
    parser.add_argument('--retrieval_model', type=str, default=f'{ROOT}/retrieval_model/runs/u9zyj9na/e27.pt')

    parser.add_argument('--food_type', type=str, default='')

    parser.add_argument('--level', type=int, default=2)
    parser.add_argument("--ca", type=int, default=1)
    return parser
