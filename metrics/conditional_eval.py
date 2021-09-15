import torch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from torchnet import meter
from torchvision import transforms

import sys
sys.path.append('../')
from mpg_plusView.train_on_pizza10 import load_mpg
from mpg_plusView.datasets import Pizza10Dataset
from ingr_classifier.train import load_classifier as load_ingr_classifier
from view_regressor.train import load_classifier as load_view_regressor
from common import infinite_loader, requires_grad, save_captioned_image
from common import normalize, resize
from metrics.calc_inception import load_patched_inception_v3
from metrics.fid import calc_fid

def make_image_caption( ingr_label, ingr_pred, view_label, view_pred):
    caption = []
    for sample_ingr_label, sample_ingr_pred, sample_view_label, sample_view_pred in zip(ingr_label, ingr_pred, view_label, view_pred):
        cap = '|'.join([f'{x:.2f}' for x in sample_ingr_label])
        cap += '\n'
        cap += '|'.join([f'{x:.2f}' for x in sample_ingr_pred])
        cap += '\n'
        cap += '|'.join([f'{x:>6.2f}' for x in sample_view_label])
        cap += '\n'
        cap += '|'.join([f'{x:>6.2f}' for x in sample_view_pred])
        caption.append(cap)
    return caption

class Attr:
    ANGLE=0
    SCALE=1
    DX=2
    DY=3

class ConditionalEvaluator():
    def __init__(self, args):
        device = args.device

        ckpt_args, _, label_encoder, _, _, netG, _, _, _ = load_mpg(args.ckpt_path)
        np.random.seed(ckpt_args.seed)
        torch.manual_seed(ckpt_args.seed)

        requires_grad(label_encoder, False)
        requires_grad(netG, False)
        label_encoder = label_encoder.eval().to(device)
        netG = netG.eval().to(device)
        self.label_encoder = label_encoder
        self.netG = netG

        _, _, ingr_classifier, _ = load_ingr_classifier(ckpt_args.ingr_classifier_path)
        requires_grad(ingr_classifier, False)
        self.ingr_classifier = ingr_classifier.eval().to(device)

        _, _, view_regressor, _ = load_view_regressor(ckpt_args.view_regressor_path)
        requires_grad(view_regressor, False) 
        self.view_regressor = view_regressor.eval().to(device)

        # load inception model
        inception = load_patched_inception_v3()
        requires_grad(inception, False)
        self.inception = inception.eval().to(device)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.dataset = Pizza10Dataset(ckpt_args.lmdb_file, size=ckpt_args.size, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
        cache_dir = os.path.dirname(ckpt_args.view_regressor_path)
        cache = os.path.join(cache_dir, 'pizza10_predictions_raw.pt')
        if os.path.exists(cache):
            print(f'load cache from {cache}')
            outputs_raw = torch.load(cache)
        else:
            outputs_raw = []
            with torch.no_grad():
                for _,img,_ in tqdm(dataloader):
                    img = img.to(device)
                    img = resize(img, size=224)
                    output = self.view_regressor(img).cpu()
                    # pdb.set_trace()
                    outputs_raw.append(output)
            outputs_raw = torch.cat(outputs_raw, dim=0)
            torch.save(outputs_raw, cache)
        self.outputs_raw = outputs_raw
        self.df = pd.read_csv(os.path.join(cache_dir, 'real_stats_for_cond_eval.csv'))
        self.scales = torch.tensor([75.0, 3.0, 112.0, 112.0])
        self.ckpt_args = ckpt_args
        self.batch_size = args.batch_size
        self.device = device
        self.fixed_noise = torch.randn(args.batch_size, ckpt_args.z_dim, device=device)
        with torch.no_grad():
            self.mean_latent = netG.mean_latent(args.truncation_mean)


    def eval_in_limit_range_for_limit_real_and_fake(self, truncation=1.0, attr_idx=Attr.ANGLE, num_eval_points=5, num_samples=100):
        name = self.df.iloc[attr_idx]['name']
        mean = self.df.iloc[attr_idx]['mean']
        std = self.df.iloc[attr_idx]['std']
        print(f'working on {name} with mean={mean:.2f} and std={std:.2f}')
        std_multiplier = self.df.iloc[attr_idx]['std_multiplier']
        delta = (2*std_multiplier)*std/num_eval_points

        file_dir = os.path.dirname(__file__)
        save_dir = os.path.join(file_dir, 'limit_range', name)
        os.makedirs(save_dir, exist_ok=True)

        stats = []
        for point in np.linspace(mean-std_multiplier*std, mean+std_multiplier*std, num=num_eval_points):
            left = point - delta/2
            right = point + delta/2
            value_dict = self.eval_one_point_in_limit_range_for_limit_real_and_fake(truncation=truncation, attr_idx=attr_idx, left=left, right=right)
            stats.append(value_dict)
        df = pd.DataFrame(stats)
        stats_path = os.path.join(save_dir, 'stats_for_cond_eval.csv')
        df.to_csv(stats_path)

    
    def eval_one_point_in_limit_range_for_limit_real_and_fake(self, truncation=1.0, attr_idx=Attr.ANGLE, left=0.0, right=25.0, num_samples=100, noise_is_same=False):
        name = self.df.iloc[attr_idx]['name']
        outputs = self.outputs_raw[:, attr_idx] * self.scales[attr_idx]
        attr_pred = outputs
        idxs = ((attr_pred>=left) & (attr_pred<right)).nonzero(as_tuple=False).squeeze(1)
        print(f'left={left:>6.2f}, right={right:>6.2f}, num_samples={idxs.shape[0]:>6d}')
        assert len(idxs) > num_samples
        subset = torch.utils.data.Subset(self.dataset, indices=idxs[:num_samples])
        loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, num_workers=8)

        file_dir = os.path.dirname(args.ckpt_path)
        save_dir = os.path.join(file_dir, 'limit_range', name)
        os.makedirs(save_dir, exist_ok=True)
        ingr_labels = []
        view_labels = []
        real_ingr_preds = []
        real_view_preds = []
        fake_ingr_preds = []
        fake_view_preds = []
        
        real_features = []
        fake_features = []
        with torch.no_grad():
            batch_idx = 0
            for txt, img, real_ingr_label in tqdm(loader):
                bs = img.shape[0]
                img = img.to(self.device)
                real_ingr_label = real_ingr_label.to(self.device)
                classifier_input_img = resize(img, size=224)
                real_ingr_pred = self.ingr_classifier(classifier_input_img)

                # dummy real_view_label
                real_view_label = torch.zeros(bs, 4).to(self.device)

                real_view_pred = self.view_regressor(classifier_input_img)
                # pdb.set_trace()
                label = torch.cat([real_ingr_label, real_view_pred], dim=1)
                if noise_is_same:
                    noise = self.fixed_noise
                else:
                    noise = torch.randn_like(self.fixed_noise)
                    noise = noise[:bs]
                txt_feat = self.label_encoder(label)
                fake, _ = self.netG(txt_feat, noise, truncation=truncation, truncation_latent=self.mean_latent)
                classifier_input_img = resize(fake, size=224)
                fake_ingr_pred = self.ingr_classifier(classifier_input_img)
                fake_view_pred = self.view_regressor(classifier_input_img)

                inception_input_img = 2*(img-img.min())/(img.max()-img.min())-1
                feat = self.inception(inception_input_img)[0].view(inception_input_img.shape[0], -1)
                real_features.append(feat.to("cpu"))
                
                inception_input_img = 2*(fake-fake.min())/(fake.max()-fake.min())-1
                feat = self.inception(inception_input_img)[0].view(inception_input_img.shape[0], -1)
                fake_features.append(feat.to("cpu"))

                # raw to human-readable
                real_view_label = real_view_label.cpu()*self.scales
                real_ingr_pred = torch.sigmoid(real_ingr_pred)
                real_view_pred = real_view_pred.cpu()*self.scales
                fake_ingr_pred = torch.sigmoid(fake_ingr_pred)
                fake_view_pred = fake_view_pred.cpu()*self.scales

                # append to all results
                ingr_labels.append(real_ingr_label)
                view_labels.append(real_view_label)
                real_ingr_preds.append(real_ingr_pred)
                real_view_preds.append(real_view_pred)
                fake_ingr_preds.append(fake_ingr_pred)
                fake_view_preds.append(fake_view_pred)

                # save image
                caption = make_image_caption(real_ingr_label, real_ingr_pred, real_view_label, real_view_pred)
                save_captioned_image(caption, img, os.path.join(save_dir, f'{name}_trunc={truncation:.2f}_[{left:>6.2f}, {right:>6.2f})_real_batch={batch_idx}.jpg'), color=(255,255,0), font=10)
                caption = make_image_caption(real_ingr_label, fake_ingr_pred, real_view_pred, fake_view_pred)
                save_captioned_image(caption, fake, os.path.join(save_dir, f'{name}_trunc={truncation:.2f}_[{left:>6.2f}, {right:>6.2f})_fake_batch={batch_idx}.jpg'), color=(255,255,0), font=10)

                batch_idx += 1

        ingr_labels = torch.cat(ingr_labels, dim=0)
        view_labels = torch.cat(view_labels, dim=0)
        real_ingr_preds = torch.cat(real_ingr_preds, dim=0)
        real_view_preds = torch.cat(real_view_preds, dim=0)
        fake_ingr_preds = torch.cat(fake_ingr_preds, dim=0)
        fake_view_preds = torch.cat(fake_view_preds, dim=0)
        real_features = torch.cat(real_features, 0).numpy()
        fake_features = torch.cat(fake_features, 0).numpy()
        
        sample_mean = np.mean(fake_features, 0)
        sample_cov = np.cov(fake_features, rowvar=False)
        real_mean = np.mean(real_features, 0)
        real_cov = np.cov(real_features, rowvar=False)
        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

        mtr = meter.APMeter()
        mtr.add(real_ingr_preds, ingr_labels)
        APs = mtr.value()
        mAP_real_ingr = APs.mean().item() # mean average precision
        
        mean_abs_err_real_view = abs(view_labels - real_view_preds).mean(dim=0).cpu().tolist()
        std_abs_err_real_view = abs(view_labels - real_view_preds).std(dim=0).cpu().tolist()
        mean_abs_err_real_angle = mean_abs_err_real_view[0]
        std_abs_err_real_angle = std_abs_err_real_view[0]
        mean_abs_err_real_scale = mean_abs_err_real_view[1]
        std_abs_err_real_scale = std_abs_err_real_view[1]
        mean_abs_err_real_dx = mean_abs_err_real_view[2]
        std_abs_err_real_dx = std_abs_err_real_view[2]
        mean_abs_err_real_dy = mean_abs_err_real_view[3]
        std_abs_err_real_dy = std_abs_err_real_view[3]


        mtr = meter.APMeter()
        mtr.add(fake_ingr_preds, ingr_labels)
        APs = mtr.value()
        mAP_fake_ingr = APs.mean().item() # mean average precision

        mean_abs_err_fake_view = abs(real_view_preds - fake_view_preds).mean(dim=0).cpu().tolist()
        std_abs_err_fake_view = abs(real_view_preds - fake_view_preds).std(dim=0).cpu().tolist()
        mean_abs_err_fake_angle = mean_abs_err_fake_view[0]
        std_abs_err_fake_angle = std_abs_err_fake_view[0]
        mean_abs_err_fake_scale = mean_abs_err_fake_view[1]
        std_abs_err_fake_scale = std_abs_err_fake_view[1]
        mean_abs_err_fake_dx = mean_abs_err_fake_view[2]
        std_abs_err_fake_dx = std_abs_err_fake_view[2]
        mean_abs_err_fake_dy = mean_abs_err_fake_view[3]
        std_abs_err_fake_dy = std_abs_err_fake_view[3]
        
        return {
            'attr': name,
            'truncation': truncation,
            'left': left,
            'right': right,
            'mAP_real_ingr': mAP_real_ingr,
            'mean_abs_err_real_angle': mean_abs_err_real_angle,
            'std_abs_err_real_angle': std_abs_err_real_angle,
            'mean_abs_err_real_scale': mean_abs_err_real_scale,
            'std_abs_err_real_scale': std_abs_err_real_scale,
            'mean_abs_err_real_dx': mean_abs_err_real_dx,
            'std_abs_err_real_dx': std_abs_err_real_dx,
            'mean_abs_err_real_dy': mean_abs_err_real_dy,
            'std_abs_err_real_dy': std_abs_err_real_dy,
            'fid': fid,
            'mAP_fake_ingr': mAP_fake_ingr,
            'mean_abs_err_fake_angle': mean_abs_err_fake_angle,
            'std_abs_err_fake_angle': std_abs_err_fake_angle,
            'mean_abs_err_fake_scale': mean_abs_err_fake_scale,
            'std_abs_err_fake_scale': std_abs_err_fake_scale,
            'mean_abs_err_fake_dx': mean_abs_err_fake_dx,
            'std_abs_err_fake_dx': std_abs_err_fake_dx,
            'mean_abs_err_fake_dy': mean_abs_err_fake_dy,
            'std_abs_err_fake_dy': std_abs_err_fake_dy,
        }

    def eval_in_full_range_for_limit_fake(self, truncation=1.0, attr_idx=Attr.ANGLE, num_eval_points=5, num_samples=5000, support='full'):
        stats = np.array([
            [0.16, 0.12],
            [0.81, 0.06],
            [0.01, 0.08],
            [-0.08, 0.08]
        ])

        if support == 'full':
            full_ranges = torch.tensor([
                [0.0, 75.0],
                [1.0, 3.0],
                [-112.0, 112.0],
                [-112.0, 112.0]
            ])
        else:
            full_ranges = torch.zeros(4, 2)
            for i in range(4):
                mu, sigma = stats[i][0], stats[i][1]
                full_ranges[i,0] = mu - 3*sigma
                full_ranges[i,1] = mu + 3*sigma
            full_ranges = full_ranges * self.scales.unsqueeze(1)

        full_range = full_ranges[attr_idx]

        def randomize_ingr_label(bs):
            ingr_label = torch.zeros(bs, 10)
            for i in range(bs):
                idxs = np.random.choice(10, np.random.randint(4), replace=False)
                ingr_label[i, idxs] = 1.0
            return ingr_label

        def randomize_view_label(bs, attr_idx, attr_val):
            view_label = torch.rand(bs, 4)
            for i in range(4):
                mu, sigma = stats[i][0], stats[i][1]
                view_label[:,i] = 3*sigma * (2*view_label[:,i]-1) + mu
            view_label[:, attr_idx] = attr_val / self.scales[attr_idx]
            return view_label

        import pickle
        name = self.df.iloc[attr_idx]['name']
        print(f'\nworking on {name}')

        real_stat_filename = 'inception_pizza10.pkl'
        print(f'load real image statistics from {real_stat_filename}')
        with open(real_stat_filename, 'rb') as f:
            embeds = pickle.load(f)
            real_mean = embeds['mean']
            real_cov = embeds['cov']

        file_dir = os.path.dirname(args.ckpt_path)
        save_dir = os.path.join(file_dir, 'full_range', f'{support}_support', name)
        os.makedirs(save_dir, exist_ok=True)
        
        bs = 50
        assert num_samples % bs == 0
        val_dicts = []
        for attr_val in np.linspace(full_range[0], full_range[1], num_eval_points):
            batch_idx = 0
            ingr_labels = []
            view_labels = []
            fake_ingr_preds = []
            fake_view_preds = []
            fake_features = []
            print(f'{name} = {attr_val:.2f}')
            for _ in tqdm(range(num_samples//bs)):
                ingr_label = randomize_ingr_label(bs).to(self.device)
                view_label = randomize_view_label(bs, attr_idx, attr_val).to(self.device)
                label = torch.cat([ingr_label, view_label], dim=1)
                z = torch.randn(bs, 256).to(self.device)
                txt_feat = self.label_encoder(label)

                if self.ckpt_args.encoder == 'simple*':
                    z = torch.cat([txt_feat[:,0], z], dim=1)
                    fake, _ = self.netG(z, truncation=truncation, truncation_latent=self.mean_latent)
                else:
                    fake, _ = self.netG(txt_feat, z, truncation=truncation, truncation_latent=self.mean_latent)

                classifier_input_img = resize(fake, size=224)
                fake_ingr_pred = self.ingr_classifier(classifier_input_img)
                fake_view_pred = self.view_regressor(classifier_input_img)

                inception_input_img = 2*(fake-fake.min())/(fake.max()-fake.min())-1
                feat = self.inception(inception_input_img)[0].view(inception_input_img.shape[0], -1)
                fake_features.append(feat.cpu())

                # raw to human-readable
                ingr_label = ingr_label.cpu()
                view_label = view_label.cpu()*self.scales
                fake_ingr_pred = torch.sigmoid(fake_ingr_pred.cpu())
                fake_view_pred = fake_view_pred.cpu()*self.scales

                # save results
                ingr_labels.append(ingr_label)
                view_labels.append(view_label)
                fake_ingr_preds.append(fake_ingr_pred)
                fake_view_preds.append(fake_view_pred)

                # save image
                if batch_idx < 2:
                    caption = make_image_caption(ingr_label, fake_ingr_pred, view_label, fake_view_pred)
                    save_captioned_image(caption, fake, os.path.join(save_dir, f'{name}_trunc={truncation:.2f}_val={attr_val:>6.2f}_fake_batch={batch_idx}.jpg'), nrow=10, color=(255,255,0), font=10)

                batch_idx += 1

            ingr_labels = torch.cat(ingr_labels, dim=0)
            view_labels = torch.cat(view_labels, dim=0)
            fake_ingr_preds = torch.cat(fake_ingr_preds, dim=0)
            fake_view_preds = torch.cat(fake_view_preds, dim=0)
            fake_features = torch.cat(fake_features, 0).numpy()
            
            sample_mean = np.mean(fake_features, 0)
            sample_cov = np.cov(fake_features, rowvar=False)
            fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

            mtr = meter.APMeter()
            mtr.add(fake_ingr_preds, ingr_labels)
            APs = mtr.value()
            mAP_fake_ingr = APs.mean().item() # mean average precision

            mean_abs_err_fake_view = abs(view_labels - fake_view_preds).mean(dim=0).cpu().tolist()
            std_abs_err_fake_view = abs(view_labels - fake_view_preds).std(dim=0).cpu().tolist()
            mean_abs_err_fake_angle = mean_abs_err_fake_view[0]
            std_abs_err_fake_angle = std_abs_err_fake_view[0]
            mean_abs_err_fake_scale = mean_abs_err_fake_view[1]
            std_abs_err_fake_scale = std_abs_err_fake_view[1]
            mean_abs_err_fake_dx = mean_abs_err_fake_view[2]
            std_abs_err_fake_dx = std_abs_err_fake_view[2]
            mean_abs_err_fake_dy = mean_abs_err_fake_view[3]
            std_abs_err_fake_dy = std_abs_err_fake_view[3]
            
            val_dict = {
                'attr': name,
                'attr_val': attr_val,
                'truncation': truncation,
                'fid': fid,
                'mAP_fake_ingr': mAP_fake_ingr,
                'mean_abs_err_fake_angle': mean_abs_err_fake_angle,
                'std_abs_err_fake_angle': std_abs_err_fake_angle,
                'mean_abs_err_fake_scale': mean_abs_err_fake_scale,
                'std_abs_err_fake_scale': std_abs_err_fake_scale,
                'mean_abs_err_fake_dx': mean_abs_err_fake_dx,
                'std_abs_err_fake_dx': std_abs_err_fake_dx,
                'mean_abs_err_fake_dy': mean_abs_err_fake_dy,
                'std_abs_err_fake_dy': std_abs_err_fake_dy,
            }
            val_dicts.append(val_dict)
        
        df = pd.DataFrame(val_dicts)
        print(df)
        stats_path = os.path.join(save_dir, 'stats_for_cond_eval.csv')
        df.to_csv(stats_path)
        return val_dicts

if __name__ == '__main__':
    from common import load_args
    args = load_args()
    
    evaluator = ConditionalEvaluator(args)

    # evaluator.eval_one_point_in_limit_range_for_limit_real_and_fake(truncation=1.0, attr_idx=Attr.DX, left=-10.60, right=-4.06, num_samples=100, noise_is_same=False)
    # for attr_idx in range(4):
    #     evaluator.eval_in_limit_range_for_limit_real_and_fake(truncation=1.0, attr_idx=attr_idx)

    for attr_idx in range(0,4):
        evaluator.eval_in_full_range_for_limit_fake(truncation=1.0, attr_idx=attr_idx, num_eval_points=10, num_samples=200, support='full')
    # evaluator.eval_in_full_range_for_limit_fake(truncation=1.0, attr_idx=1, num_eval_points=10, num_samples=5000)
    print('Done')