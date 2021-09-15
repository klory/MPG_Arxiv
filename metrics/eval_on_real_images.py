import torch
import os
from torchnet import meter

import sys
sys.path.append('../')
from datasets.datasets import Pizza50
from ingr_classifier.train import load_classifier as load_ingr_classifier
from view_regressor.train import load_classifier as load_view_regressor
from metrics.conditional_eval import make_image_caption
import common


device = 'cuda'
ingr_classifier_path = f'{common.ROOT}/ingr_classifier/runs/1t5xrvwx/batch5000.ckpt'
view_regressor_path = f'{common.ROOT}/view_regressor/runs_view_point/1tbpdsqj/00004999.ckpt'

_, _, ingr_classifier, _ = load_ingr_classifier(ingr_classifier_path)
common.requires_grad(ingr_classifier, False)
ingr_classifier = ingr_classifier.eval().to(device)
_, _, view_regressor, _ = load_view_regressor(view_regressor_path)
view_regressor = view_regressor.eval().to(device)
common.requires_grad(view_regressor, False)

view_attr_names = ['angle', 'scale', 'dx', 'dy']
scales = torch.tensor([75.0, 3.0, 112.0, 112.0])

dataset = Pizza50(lmdb_file=f'{common.ROOT}/data/Pizza3D_lmdb', size=224)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False)
imgs = []
ingr_labels = []
ingr_preds = []
view_labels = []
view_preds = []
for img, ingr_label, view_label in loader:
    img = img.to(device)
    ingr_pred = torch.sigmoid(ingr_classifier(img).cpu())
    view_pred = view_regressor(img).cpu() * scales
    
    imgs.append(img.to(device))
    ingr_labels.append(ingr_label)
    ingr_preds.append(ingr_pred)
    view_labels.append(view_label)
    view_preds.append(view_pred)

imgs = torch.cat(imgs, dim=0)
ingr_labels = torch.cat(ingr_labels, dim=0)
ingr_preds = torch.cat(ingr_preds, dim=0)
view_labels = torch.cat(view_labels, dim=0)
view_preds = torch.cat(view_preds, dim=0)

mtr = meter.APMeter()
mtr.add(ingr_preds, ingr_labels)
APs = mtr.value()
mAP_ingr = APs.mean().item() # mean average precision
print(f'ingr mAP = {mAP_ingr:.4f}')

mean_abs_err_view = abs(view_labels - view_preds).mean(dim=0).cpu().tolist()
std_abs_err_view = abs(view_labels - view_preds).std(dim=0).cpu().tolist()
for attr, mean, std in zip(view_attr_names, mean_abs_err_view, std_abs_err_view):
    print(f'{attr} abs error = {mean:.4f} ({std:.4f})')

captions = make_image_caption(ingr_labels, ingr_preds, view_labels, view_preds)
save_dir = os.path.dirname(os.path.abspath(__file__))
common.save_captioned_image(captions, imgs, os.path.join(save_dir, f'classifier_performances.jpg'), color=(255,255,0), font=8, nrow=10)
