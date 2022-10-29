from pathlib import Path

import torch
import wandb
from PIL import Image
from filelock import FileLock
from torchvision import transforms

MODEL_NAME = 'mnist-model:latest'
MODEL_PATH = '/tmp/model'
MODEL_LOCK = '.lock-file'
PROJECT_NAME = 'ml-in-production'


def load_from_registry(model_name, model_path):
    with wandb.init(project=PROJECT_NAME) as run:
        artifact = run.use_artifact(model_name, type='model')
        artifact_dir = artifact.download(root=model_path)
        print(artifact_dir)


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Predictor:
    def __init__(self, model_path: str, device=None):
        device = device if device is not None else get_device()
        self.device = device

        self.model = torch.jit.load(Path(model_path) / 'model.pt', map_location=self.device)

        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.size = 28

    @classmethod
    def default_from_model_registry(cls) -> 'Predictor':
        with FileLock(MODEL_LOCK):
            if not (Path(MODEL_PATH) / 'model.pt').exists():
                load_from_registry(MODEL_NAME, MODEL_PATH)

        return cls(MODEL_PATH)

    def predict(self, image: Image) -> int:
        x = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(x)
        output = torch.argmax(output)
        return output.item()
