import torch.cuda
import yaml


def get_config(path):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)

    return conf


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
