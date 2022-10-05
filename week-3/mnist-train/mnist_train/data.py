from torchvision import transforms, datasets
from pathlib import Path


def load_data(path_save: Path, norm1: float = 0.1307, norm2: float = 0.3081):
    path_save = str(path_save)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm1,), (norm2,))
    ])
    dataset_train = datasets.MNIST(path_save, train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(path_save, train=False, transform=transform)

    return dataset_train, dataset_test
