import yaml
from types import SimpleNamespace
import argparse
import sys
sys.path.append('../')
from common import ROOT

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=f'{ROOT}/metrics/configs/pizza10+mpg.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        args = SimpleNamespace(**data)
    return args

