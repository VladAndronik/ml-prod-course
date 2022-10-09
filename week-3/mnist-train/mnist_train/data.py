from pathlib import Path

from torchvision import transforms

from .dataset.mnist import MNIST


# todo: use internal dataset for train


def load_data(path_save: Path, norm1: float = 0.1307, norm2: float = 0.3081, n_samples: int = 0):
    path_save = str(path_save)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm1,), (norm2,))
    ])
    # dataset_train = datasets.MNIST(path_save, train=True, download=True, transform=transform)
    dataset_train = MNIST(path_save, train=True, download=True, transform=transform, n_samples=n_samples)
    # dataset_test = datasets.MNIST(path_save, train=False, transform=transform)
    dataset_test = MNIST(path_save, train=False, transform=transform)

    return dataset_train, dataset_test
