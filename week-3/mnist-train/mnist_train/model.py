import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path

from mnist_train.networks import Net


class Model:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.root = Path(__file__).parent.parent

        self.model = Net().to(device).train().requires_grad_(True)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=config['train_params']['lr'])
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=config['train_params']['gamma'])

    def step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        output = self.model(x)
        loss = F.nll_loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses = {
            'nll': loss.item()
        }

        return losses

    def evaluate(self, dataloader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)
        accuracy = 100. * correct / len(dataloader.dataset)
        self.model.train()
        output = {
            'nll': test_loss,
            'accuracy': accuracy
        }
        return output

    def save(self, path):
        torch.save(self.model.state_dict(), path / 'model.pth')

    def create_model_card(self):
        s = f"""
            # {self.config['model_card']['project_name']}
            ## Setup
            ```
            pip install -r requirements.txt
            ```
            
            ## Develop
            ```
            export PYTHONPATH=.
            export WANDB_PROJECT={self.config['model_card']['project_name']}
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
        """
        s = '\n'.join(map(str.strip, s.split('\n')))
        with open(self.root / 'README.md', 'w') as f:
            f.write(s)


if __name__ == '__main__':
    from mnist_train.utils import get_config
    config = get_config('../conf/config.yaml')
    device = 'cpu'
    model = Model(config, device)
    model.create_model_card()
