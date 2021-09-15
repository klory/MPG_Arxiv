import torch
from tqdm import tqdm
import os
import csv
from glob import glob
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from retrieval_model.utils import rank
from retrieval_model.train import load_retrieval_model
from common import requires_grad, normalize, resize, ROOT

@torch.no_grad()
def compute_medR(args):
    ckpt_path = args.ckpt_path
    print(f'working on {ckpt_path}')
    batch_generator = BatchGenerator(args)
    txt_outputs = []
    img_outputs = []
    for _ in tqdm(range(1000//args.batch_size)):
        # generate
        txt, fake_img = batch_generator.generate_MedR()
        # fake_img: normalize
        fake_img = normalize(fake_img)
        # fake_img: resize
        fake_img = resize(fake_img, size=224)
        # retrieve
        txt_inputs = tokenizer(txt, truncation=True, padding=True, return_tensors="pt").to(device)
        txt_output, _ = txt_encoder(**txt_inputs)
        img_output = img_encoder(fake_img)
        txt_outputs.append(txt_output.detach().cpu())
        img_outputs.append(img_output.detach().cpu())
    txt_outputs = torch.cat(txt_outputs, dim=0).numpy()
    img_outputs = torch.cat(img_outputs, dim=0).numpy()
    retrieved_range = min(txt_outputs.shape[0], 1000)
    medR, medR_std, recalls = rank(
        txt_outputs, img_outputs, epoch=0, retrieved_type='image', 
        retrieved_range=retrieved_range, verbose=True)
    return medR

if __name__ == '__main__':
    from metrics.utils import load_args
    args = load_args()

    # assertations
    assert 'dataset_name' in args.__dict__
    assert 'ckpt_dir' in args.__dict__
    assert 'retrieval_model' in args.__dict__
    assert 'device' in args.__dict__
    assert 'batch_size' in args.__dict__

    if 'stackgan2/' in args.ckpt_dir:
        from stackgan2.generate_batch import BatchGenerator
    elif 'AttnGAN/' in args.ckpt_dir:
        from AttnGAN.code.generate_batch_Attn import BatchGenerator
    elif 'mpg/' in args.ckpt_dir:
        from mpg.generate_batch import BatchGenerator
    
    device = args.device
    args_ret, _, tokenizer, txt_encoder, img_encoder, _ = load_retrieval_model(ROOT / args.retrieval_model, device)
    requires_grad(txt_encoder, False)
    requires_grad(img_encoder, False)
    txt_encoder = txt_encoder.eval()
    img_encoder = img_encoder.eval()


    # *******************************
    # only run for one checkpoint
    # ********************************
    if hasattr(args, 'ckpt_path'):
        medR = compute_medR(args)
        print(f'MedR={medR}')
        sys.exit(0)


    # *******************************
    # run for all checkpoints
    # ********************************
    filename = os.path.join(args.ckpt_dir, 'medR.csv')

    # load values that are already computed
    computed = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                computed += [row[0]]
    
    # prepare to write
    f = open(filename, mode='a')
    writer = csv.writer(f, delimiter=',')

    # find checkpoints
    ckpt_paths = glob(os.path.join(args.ckpt_dir, '*.ckpt')) + glob(os.path.join(args.ckpt_dir, '*.pt'))+glob(os.path.join(args.ckpt_dir, '*.pth'))
    ckpt_paths = sorted(ckpt_paths)
    print('records:', ckpt_paths)
    print('computed:', computed)
    for ckpt_path in ckpt_paths:
        print()
        iteration = os.path.basename(ckpt_path).split('.')[0]
        if iteration in computed:
            print('already computed')
            continue

        args.ckpt_path = ckpt_path
        medR = compute_medR(args)
        print(f'{iteration}, MedR={medR}')
        writer.writerow ([iteration, medR])

    f.close()
    medRs = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            medR = float(row[1])
            medRs += [medR]
    fig = plt.figure(figsize=(6,6))
    plt.plot(medRs)
    plt.savefig(os.path.join(args.ckpt_dir, 'medR.png'))