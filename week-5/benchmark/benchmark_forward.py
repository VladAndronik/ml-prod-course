import time

import torch
from serving.predictor import get_device, load_from_registry
from pathlib import Path
import GPUtil


# device = get_device()
device = 'cpu'
root = Path(__file__).parent
# model_path = '/tmp/model/model.pt'
model_path = root / 'weights/model_static_quantized.pt'
model_name = 'mnist-model:latest'


def main(batch_size=4, half=False):

    # if not Path(model_path).exists():
    #     load_from_registry(model_name, Path(model_path).parent)

    data = torch.rand(batch_size, 1, 28, 28).to(device)
    if half:
        data = data.half()

    memory_before_init = GPUtil.getGPUs()[0].memoryUsed
    model = torch.jit.load(model_path, map_location=device)
    if half:
        model = model.half()
    memory_after_init = GPUtil.getGPUs()[0].memoryUsed

    # assert not next(model.parameters()).requires_grad

    times = []
    for _ in range(110):
        time1 = time.time()
        _ = model(data)
        time1 = time.time() - time1
        times.append(time1)
    memory_after_forward = GPUtil.getGPUs()[0].memoryUsed

    print(f"Batch Size: {batch_size} | Use Half: {half} | GPU Memory Init: {memory_after_init - memory_before_init} | "
          f"GPU Memory overall: {memory_after_forward - memory_before_init}")
    print(f"Total speed: {1 / (sum(times[10:]) / (len(times) - 10) / batch_size)} FPS")


if __name__ == '__main__':
    main(batch_size=4, half=False)
