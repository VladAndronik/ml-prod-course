from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .dataset.mnist import MNIST
from .networks import Net
from serving.predictor import get_device


class Predictor:
    def __init__(self, device, checkpoint_path: Path, use_jit = False, train: bool = False, batch_size: int = 64):
        if use_jit:
            self.model = torch.jit.load(checkpoint_path, map_location=device)
        else:
            self.model = Net().to(device).eval().requires_grad_(False)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)

        self.device = device
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        path_data = Path(__file__).parent.parent / 'data'
        dataset = MNIST(path_data, train=train, transform=transform, n_samples=0, download=True)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    def run(self):
        test_loss = 0
        correct = 0
        for data, target in tqdm(self.dataloader, total=len(self.dataloader)):
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


def evaluate(checkpoint_path: Path, device: str, use_jit: bool):
    device = get_device() if device is None else device
    predictor = Predictor(device, checkpoint_path=checkpoint_path, use_jit=use_jit)
    outputs = predictor.run()
    print(outputs)
