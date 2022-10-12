from pathlib import Path

import torch
import tqdm
import wandb
from mnist_train.data import load_data
from mnist_train.model import Model
from mnist_train.utils import get_config, get_device, seed_all

root = Path(__file__).parent.parent
path_config = root / 'conf/config.yaml'
device = get_device()


def init_dirs(config):
    (root / config['dir_data']).mkdir(parents=True, exist_ok=True)
    (root / config['dir_log']).mkdir(parents=True, exist_ok=True)
    (root / config['dir_log'] / config['dir_weights']).mkdir(parents=True, exist_ok=True)


def init_loaders(config):
    dir_data = root / config['dir_data']
    dataloaders_kwargs = {
        'num_workers': config['data']['num_workers'],
        'shuffle': config['data']['shuffle'],
        'pin_memory': config['data']['pin_memory'],
        'batch_size': config['train_params']['batch_size']
    }

    datasets_kwargs = {'norm1': config['data']['normalize_mean'],
                       'norm2': config['data']['normalize_std'],
                       'n_samples': config['data']['n_samples']
                       }

    dataset_train, dataset_test = load_data(dir_data, **datasets_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train, **dataloaders_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **dataloaders_kwargs)

    return train_loader, test_loader


def train(config_path: Path):
    config = get_config(config_path)
    seed_all(config['seed'])
    wandb.init(project=config['model_card']['project_name'], entity="vldrnk", name='test-run')
    wandb.config = config

    init_dirs(config)
    train_loader, test_loader = init_loaders(config)
    model = Model(config=config, device=device)

    for _ in tqdm.tqdm(range(config['train_params']['epochs'])):
        for batch_idx, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            output = model.step(data)

            if batch_idx % config['logs']['log_train_loss'] == 0:
                wandb.log({'train/nll': output['nll']})

            if batch_idx % config['logs']['log_test_loss'] == 0:
                output = model.evaluate(test_loader)
                for key, value in output.items():
                    wandb.log({f'test/{key}': value})

            if batch_idx % config['logs']['log_weights'] == 0:
                model.save(root / config['dir_log'] / config['dir_weights'])

        model.save(root / config['dir_log'] / config['dir_weights'])

    model.create_model_card()


if __name__ == '__main__':
    train(path_config)
