# Setup Environment
**can NOT run on CPU**

```
conda create -n mpg python=3.8
conda activate mpg
git clone git@github.com:klory/MPG_Arxiv.git
cd MPG_Arxiv
pip install -r requirements.txt
pip install git+https://github.com/pytorch/tnt.git@master
```

# Pretrained models
Pretrained models are stored in [google-link](https://drive.google.com/drive/folders/12plZIczJJcGMD8W0VnVocYh-jXwg3t8N?usp=sharing), files are already in their desired locations, so following the same directory structure will minimize burdens to run the code inside the project (some files are not necessary for the current version of the project as of 2021-03-31).

# Pizza10 dataset
Download Pizza10 Dataset from `google-link/data/Pizza10`.

## Validate

cd to `datasets/`

```
$ python datasets.py
```

# Ingredient classifier

cd to `ingr_classifier/`,

## Train
```
$ CUDA_VISIBLE_DEVICES=0 python from train.py --wandb=0
```

## Validate
Download the pretrained model `google-link/ingr_classifier/runs/pizza10/1t5xrvwx/batch5000.ckpt`:
```
$ CUDA_VISIBLE_DEVICES=0 python val.py --ckpt_path=runs/1t5xrvwx/batch5000.ckpt
```

# Ingredient retrieval model

cd to `retrieval_model/`,

## Datasets
The datset is designed to make sure every sample contains different ingredient set, so the size of the dataset is the number of unique ingredient sets in `Pizza10` (length=273).

```
$ python datasets_ret.py
```

## Models 
```
$ CUDA_VISIBLE_DEVICES=0 python models.py
```

## Train
```
CUDA_VISIBLE_DEVICES=0 python train.py --wandb=0
```

## Validate
Download the pretrained model `google-link/retrieval_model/runs/u9zyj9na/e27.pt`:

```
$ CUDA_VISIBLE_DEVICES=0 python val.py --ckpt_path=runs/u9zyj9na/e27.pt
```

# MPG
| [Paper](https://arxiv.org/abs/2012.02821) | [Interactive demo](http://foodai.cs.rutgers.edu:2021/) | [Video](https://youtu.be/x3XKXMd1oC8) |

cd to `mpg/`,

## Models
```
$ CUDA_VISIBLE_DEVICES=0 python models.py
```

## Train

Assume you already have the pretrained ingredient classifier.

```
$ CUDA_VISIBLE_DEVICES=0,1 python train.py --wandb=0
...
```

## Validate
Download the pretrained model `google-link/mpg/runs/qo6nm72k/280000.pt`.

cd to `metrics/`:

```
CUDA_VISIBLE_DEVICES=0 python generate_samples.py --model=mpg
```

# StackGAN2
cd to `stackgan2/`,

## Models
```
CUDA_VISIBLE_DEVICES=0 python models.py
```

## Train
```
CUDA_VISIBLE_DEVICES=0 python models.py --cycle_img=0.0
```

## Validate
cd to `metrics/`

```
CUDA_VISIBLE_DEVICES=0 python generate_samples.py --model=mpg
```

# CookGAN

Check the [official code of CookGAN](https://github.com/klory/CookGAN) for more details.

## Train
```
CUDA_VISIBLE_DEVICES=0 python models.py --cycle_img=1.0
```

## Validate
cd to `metrics/`

```
CUDA_VISIBLE_DEVICES=0 python generate_samples.py --model=mpg
```

# AttnGAN (canary)

## Train
TODO

## Validate
cd to `metrics/`

```
CUDA_VISIBLE_DEVICES=0 python generate_samples.py --model=mpg
```

# Metrics
> cd to `metrics/`,

## FID (Frechet Inception Distance)
To compute FID, we need to first compute the statistics of the real images.

```
CUDA_VISIBLE_DEVICES=0 python calc_inception.py
```

```
$ CUDA_VISIBLE_DEVICES=0 python fid.py --model=mpg
```

## MedR (Medium Rank) for Ingredients
Computing MedR utilizes the pre-trained retrieval model.

```
$ CUDA_VISIBLE_DEVICES=0 python medR.py --model=mpg
```

## mAP (mean Average Precision) for Ingredients
Computing mAP utilizes the pre-trained ingredient classifier.

```
$ CUDA_VISIBLE_DEVICES=0 python mAP.py --model=mpg
```

# Interactive Web Demo (canary)

cd to `metrics/`.


```
CUDA_VISIBLE_DEVICES=0 streamlit run app.py
```