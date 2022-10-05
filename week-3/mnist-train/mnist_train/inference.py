from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from .networks import Net
from .utils import get_config


class Predictor:
    def __init__(self, device, config_path: Path, checkpoint_path: Path):
        self.config = get_config(config_path)
        self.model = Net().to(device).train().requires_grad_(True)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
        self.device = device
        self.size = self.config['image_size']

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.config['data']['normalize_mean'],), (self.config['data']['normalize_std'],))
        ])
        path_data = Path(__file__).parent.parent / self.config['dir_data']
        dataset = datasets.MNIST(path_data, train=False, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)

    def run(self):
        test_loss = 0
        correct = 0
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataloader.dataset)
        accuracy = 100. * correct / len(self.dataloader.dataset)
        output = {
            'nll': test_loss,
            'accuracy': accuracy
        }
        return output


def evaluate(checkpoint_path: Path, config_path: Path):
    device = 'cuda:0'
    predictor = Predictor(device, config_path=config_path, checkpoint_path=checkpoint_path)
    outputs = predictor.run()
    print(outputs)
