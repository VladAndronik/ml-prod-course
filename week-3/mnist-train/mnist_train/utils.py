import torch.cuda
import yaml
import torch
import numpy as np
import random


def seed_all(seed):
    if not seed:
        seed = 17

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config(path):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)

    return conf


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
