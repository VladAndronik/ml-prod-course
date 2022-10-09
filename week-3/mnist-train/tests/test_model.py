from pathlib import Path

from mnist_train.inference import Predictor
from mnist_train.train import train
from mnist_train.utils import get_config, get_device

# todo: test on overfit one batch
# todo: test on running train on small data - check that checkpoints and summaring works end-to-end

# optional after adding other stuff:
# todo: min func. test that easy images works for trained model (where to take the trained model?)
# todo: adding some noise to photo does not change the output
# todo: directional: cutout of an image leads to change in label (might be just random)
# todo: add ci for all of this

root = Path(__file__).parent.parent


def test_overfit_on_batch():
    path_config = root / 'tests/conf/test_overfit_config.yaml'
    config = get_config(path_config)
    device = get_device()
    train(path_config)

    predictor = Predictor(device, path_config, root / config['dir_log'] / config['dir_weights'] / 'model.pth', batch_size=4,
                          train=True)
    output = predictor.run()
    assert output['nll'] < 1e-2
    assert output['accuracy'] > 95


def test_pipeline():
    path_config = root / 'tests/conf/test_pipeline_config.yaml'

    config = get_config(path_config)
    train(path_config)

    dir_data = root / config['dir_data']
    dir_ckpt = root / config['dir_log']
    dir_weight = root / config['dir_log'] / config['dir_weights']
    path_model_card = root / 'README.md'

    assert dir_data.exists(), 'Data dir does not exist'
    assert dir_ckpt.exists(), 'Logs dir does not exist'
    assert dir_weight.exists(), 'Weights dir does not exist'
    assert path_model_card.exists(), 'Model card was not created'
