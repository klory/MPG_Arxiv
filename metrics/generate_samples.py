import torch
import os
import pdb
import numpy as np
import sys
sys.path.append('../')
import common


def randomize_ingr_label(bs):
    ingr_label = torch.zeros(bs, 10)
    for i in range(bs):
        idxs = np.random.choice(10, np.random.randint(4), replace=False)
        ingr_label[i, idxs] = 1.0
    return ingr_label


if __name__ == '__main__':
    from metrics.utils import load_args
    args = load_args()
    
    # assertations
    assert 'ckpt_path' in args.__dict__
    assert 'device' in args.__dict__
    assert 'batch_size' in args.__dict__
    assert 'model_name' in args.__dict__

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    print('current file dir:', cur_dir)
    if 'stackgan2/' in args.ckpt_dir:
        from stackgan2.generate_batch import BatchGenerator
        os.chdir('../stackgan2/')
    elif 'AttnGAN/' in args.ckpt_dir:
        from AttnGAN.code.generate_batch_Attn import BatchGenerator
        os.chdir('../AttnGAN/code/')
    elif 'mpg/' in args.ckpt_dir:
        assert 'truncation' in args.__dict__
        from mpg.generate_batch import BatchGenerator
        os.chdir('../mpg/')

    device = args.device
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_generator = BatchGenerator(args)
    os.chdir(cur_dir)

    save_dir = f'outputs/seed={seed}'
    os.makedirs(save_dir, exist_ok=True)
    
    # ****************************************************************
    # Generate some images
    # ****************************************************************

    print('generating images...')
    txt, real, fake, label = batch_generator.generate_all()
    fp = f'{save_dir}/{args.model_name}_trunc={args.truncation:.2f}.png'
    common.save_captioned_image(txt, fake, fp, font=15, opacity=0.2, color=(255,255,0), loc=(0,0), nrow=int(np.sqrt(args.batch_size)), pad_value=1)
    print(f'saved to {fp}')