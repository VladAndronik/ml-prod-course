from benchmark.inference import evaluate
from pathlib import Path
from benchmark.networks import NetQ as Net
from serving.predictor import get_device
import torch
import os

checkpoint_path = Path(__file__).parent / 'weights/model_baseline.pth'
# device = get_device()
device = 'cpu'


def run(path):
    evaluate(path, device, use_jit=True)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def load_model():
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.requires_grad_(False).eval()

    return model


def quantize(save_path):
    backend = "qnnpack"

    model = load_model()

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    print_size_of_model(model_static_quantized)
    model_jit = torch.jit.script(model_static_quantized, torch.rand(1, 1, 28, 28))
    torch.jit.save(model_jit, save_path)


if __name__ == '__main__':
    quantize('weights/model_static_quantized.pt')
