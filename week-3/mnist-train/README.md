
# ml-in-production
## Setup
```
pip install -r requirements.txt
```

## Develop
```
export PYTHONPATH=.
export WANDB_PROJECT=ml-in-production
export WANDB_API_KEY=****************
```

## Run everything
```
python mnist_train/cli.py load-data ./data
python mnist_train/cli.py train conf/config.yaml
python mnist_train/cli.py evaluate logs/weights/checkpoint_stage1/model.pth conf/config.yaml
```
## Tests
```
make test_all
```
