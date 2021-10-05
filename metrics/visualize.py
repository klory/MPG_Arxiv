import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from common import normalize, resize
from mpg.train import load_mpg
from stackgan2.train import load_stackgan2_model
from retrieval_model.train import load_retrieval_model, compute_txt_feat
from ingr_classifier.train import load_classifier
from AttnGAN.code.generate_batch_Attn import build_models as build_models_Attn
from AttnGAN.code.generate_cs import generate_one as generate_one_Attn
from common import requires_grad, ROOT

@st.cache()
def load_models(ckpt_path, device):
    if 'mpg' in ckpt_path:
        ckpt_args, _, label_encoder, _, _, g_ema, _, _, _ = load_mpg(ckpt_path, device=device)
        requires_grad(label_encoder, False)
        requires_grad(g_ema, False)
        return label_encoder.eval(), g_ema.eval()
    elif 'stackgan2' in ckpt_path:
        ckpt_args, _, netG, _, _, _ = load_stackgan2_model(ckpt_path, device)
        _, _, tokenizer, txt_encoder, _, _ = load_retrieval_model(f'{ROOT}/retrieval_model/runs/u9zyj9na/e27.pt', device)
        requires_grad(netG, False)
        requires_grad(txt_encoder, False)
        return (tokenizer, txt_encoder.eval()), netG.eval()
    elif 'AttnGAN' in ckpt_path:
        text_encoder, _, netG, _ = build_models_Attn(ckpt_path,f"{ROOT}/AttnGAN/output/food_DAMSM_2020_11_10_10_36_03/Model/text_encoder150.pth")
        requires_grad(text_encoder, False)
        requires_grad(netG, False)
        netG.eval()
        return text_encoder, netG

@st.cache()
def load_clf(ckpt_path=f'{ROOT}/ingr_classifier/runs/pizza10/1t5xrvwx/batch5000.ckpt', device='cuda'):
    _, _, classifier, _ = load_classifier(ckpt_path)
    classifier = classifier.eval().to(device)
    return classifier


def compute_mean_latent(g_ema, truncation, truncation_mean):
    if truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(truncation_mean)
    else:
        mean_latent = None
    return mean_latent

@st.cache()
def load_categories(filename):
    with open(filename, 'r') as f:
        categories = f.read().strip().split('\n')
    return categories

def truncate():
    return st.sidebar.number_input('Diversity Level', min_value=0.0, max_value=1.0, value=1.0, step=0.1)


def load_noise(noise_dim, device='cuda'):
    st.sidebar.header('Noise')
    noise = []
    for i in range(noise_dim):
        val = st.sidebar.slider(f'Dimension{i+1}', min_value=-3.0, max_value=3.0, value=torch.randn(1).item(), step=0.01)
        noise.append(val)
    noise = torch.FloatTensor(noise).unsqueeze(0).to(device)
    return noise

def to_numpy_img(tensor):
    npimg = tensor.detach().cpu().numpy()
    npimg = (npimg-npimg.min()) / (npimg.max()-npimg.min())
    return np.transpose(npimg, (1,2,0))

@torch.no_grad()
def generate_cs(noise, binary_label, label_encoder, g_ema, truncation, mean_latent):
    text_outputs = label_encoder(binary_label)
    sample, _ = g_ema(
        [noise], text_outputs, truncation=truncation, truncation_latent=mean_latent
    )
    return sample

@torch.no_grad()
def generate_cat_z(noise, binary_label, g_ema, truncation, mean_latent):
    sample, _ = g_ema(
        noise, binary_label, truncation=truncation, truncation_latent=mean_latent
    )
    return sample

@torch.no_grad()
def generate_stackgan2(noise, binary_label, tokenizer, txt_encoder, netG, i2w, device='cuda'):
    binary_label = binary_label.cpu().numpy()
    txt = ['\n'.join([i2w[idx] for idx in binary_label.nonzero()[0]])]
    txt_feat = compute_txt_feat(txt, tokenizer, txt_encoder, device=device)
    print(noise.shape, txt_feat.shape)
    fakes, _, _ = netG(noise, txt_feat)
    sample = fakes[-1]
    return sample

@torch.no_grad()
def generate_AttnGAN(noise, binary_label,txt_encoder, netG, i2w):
    pass

class Model:
    def __init__(self, name, ckpt_path):
        self.name = name
        self.ckpt_path = ckpt_path

def main():
    st.title("Pizza Generators")

    models = [
        Model(name='MPG: Multi-ingredient Pizza Generator', ckpt_path=f'{ROOT}/mpg/runs/2y8walj3/040000.pt'),
        Model(name='StackGAN2', ckpt_path=f'{ROOT}/stackgan2/runs/199iwv2f/batch080000.ckpt'),
        Model(name='CookGAN', ckpt_path=f'{ROOT}/stackgan2/runs/1q7grcoo/batch080000.ckpt'),
        # Model(name='AttnGAN', ckpt_path=f'{ROOT}/AttnGAN/output/food_attn2_2020_11_10_11_51_01/Model/netG/netG_epoch_1800.pth'),
        Model(name='MPG-mapping', ckpt_path=f'{ROOT}/mpg/runs/3ae90gw9/040000.pt'),
        Model(name='MPG-SLE', ckpt_path=f'{ROOT}/mpg/runs/c1ysmsbe/040000.pt'),
        Model(name='MPG-SLE*', ckpt_path=f'{ROOT}/mpg/runs/306bmus9/040000.pt'),
        Model(name='MPG-CR', ckpt_path=f'{ROOT}/mpg/runs/1zg4fv6o/040000.pt'),
        # Model(name='MPG-matching', ckpt_path=f'{ROOT}/mpg/runs/qo6nm72k/040000.pt'),
        Model(name='MPG-uncond', ckpt_path=f'{ROOT}/mpg/runs/2ftkg1ww/030000.pt'),
    ]
    model_names = [x.name for x in models]
    model_ckpt_paths = [x.ckpt_path for x in models]

    categories_filename = f'{ROOT}/data/Pizza10/categories.txt'
    device='cuda'
    bs = 1

    option = st.selectbox(
        'Which model to visualize?',
        model_names,
        index=0
    )

    ckpt_path = model_ckpt_paths[model_names.index(option)]
    st.write(ckpt_path)

    if st.sidebar.button('Refresh'):
        st.sidebar.write('new image generated')

    encoder, generator = load_models(ckpt_path, device)

    categories = load_categories(categories_filename)
    binary_label = torch.zeros(bs,10).to(device)
    for i, category in enumerate(categories):
        val = st.sidebar.checkbox(category)
        binary_label[:,i] = float(val)

    if 'mpg' in ckpt_path:
        label_encoder = encoder
        truncation = truncate()
        truncation_mean = 4096
        mean_latent = compute_mean_latent(generator, truncation, truncation_mean)
        noise = load_noise(256, device)
        sample = generate_cs(noise, binary_label, label_encoder, generator, truncation, mean_latent)
    elif 'stackgan2' in ckpt_path:
        tokenizer, txt_encoder = encoder
        noise = load_noise(100, device)
        i2w = {i:w for i,w in enumerate(categories)}
        sample = generate_stackgan2(noise, binary_label, tokenizer, txt_encoder, generator, i2w, device=device)
    elif 'AttnGAN' in ckpt_path:
        noise = load_noise(100, device)
        sample = generate_one_Attn(noise,binary_label, encoder, generator)

    if sample.ndim == 3:
        sample = sample.unsqueeze(0)
    
    img_np = to_numpy_img(sample.squeeze(0))
    st.image(img_np)

    classifier = load_clf(device='cuda')
    requires_grad(classifier, False)
    fake_img = normalize(sample)
    fake_img = resize(fake_img, size=224)
    output = classifier(fake_img)
    probs = torch.sigmoid(output).squeeze().cpu().numpy()
    gt = binary_label.detach().cpu().squeeze().numpy()
    fig, ax = plt.subplots()
    ax.title.set_text('Ingredient classifier for synthetic image')
    ind = np.arange(len(probs))
    width = 0.3
    ax.barh(ind, probs, width, color='green', label='Prediction')
    ax.barh(ind + width, gt, width, color='red', label='Ground Truth')
    ax.set(yticks=ind + width, yticklabels=categories)
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
