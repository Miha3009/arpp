import torch
import torch.nn as nn
from pathlib import Path
import signal
import pandas as pd
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from saver import load_from_config

class Experiment:
    def __init__(self, name, dataset=None, model=None, metric=None, optimizer=None,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        base_dir = '../experiments',
        checkpoint = None
    ):
        self.name = name
        self.device = device

        self.exp_dir = Path(base_dir) / name
        self.checkpoints_dir = self.exp_dir / 'checkpoints'
        self.images_dir = self.exp_dir / 'images'

        if self.exp_dir.exists():
            self._load_config(checkpoint or 'last')
        else:
            for d in [self.exp_dir, self.checkpoints_dir, self.images_dir]:
                d.mkdir(parents=True, exist_ok=True)

            self.dataset = dataset
            self.model = model.to(device)
            self.metric = metric
            self.optimizer = optimizer
            self.optimizer.setup(model.parameters())

            self.current_epoch = 0
            self.current_batch = 0
            self.current_loss = 0
            self.current_mode = 'train'
            self.history = []
            self._save_config()

        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print('\nОбучение прерывается...')
        self.should_stop = True

    def _save_config(self):
        self.dataset.save_config(self.exp_dir / 'dataset.json')
        self.model.save_config(self.exp_dir / 'model.json')
        self.optimizer.save_config(self.exp_dir / 'optimizer.json')
        self.metric.save_config(self.exp_dir / 'metric.json')
        self.save_checkpoint('last')

    def _load_config(self, checkpoint='last'):
        self.dataset = load_from_config(self.exp_dir / 'dataset.json')
        self.model = load_from_config(self.exp_dir / 'model.json')
        self.optimizer = load_from_config(self.exp_dir / 'optimizer.json')
        self.metric = load_from_config(self.exp_dir / 'metric.json')
        self.load_checkpoint(checkpoint)

    def save_checkpoint(self, name):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_mode': self.current_mode,
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'current_loss': self.current_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoints_dir / f'{name}.pt')
        pd.DataFrame(self.history).to_csv(self.exp_dir / 'loss_history.csv', index=False)

    def load_checkpoint(self, name: str, load_optimizer: bool = True):
        checkpoint_path = self.checkpoints_dir / f'{name}.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.optimizer.setup(self.model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_mode = checkpoint['current_mode']
        self.current_epoch = checkpoint['current_epoch']
        self.current_batch = checkpoint['current_batch']
        self.current_loss = checkpoint['current_loss']
        self.history = checkpoint['history']

    def run(self, epochs, save_every_n_epochs=5):
        while self.current_epoch < epochs:
            self.dataset.loader.set_skip(self.current_batch)
            if self.current_mode == 'train':
                self._run_train()
                self.current_mode = 'test'
            else:
                self._run_test()
                self.plot_history()
                self.current_mode = 'train'
                self.current_epoch += 1
            self.current_batch = 0
            self.current_loss = 0
            if self.current_mode == 'train' and self.current_epoch % save_every_n_epochs == 0 and self.current_epoch < epochs:
                self.save_checkpoint(f'e{self.current_epoch:03d}')
        self.save_checkpoint(f'last')

    def _run_train(self):
        self.model.train()
        self.dataset.set_mode('train')
        self.dataset.set_seed(self.current_epoch)
        pbar = tqdm(self.dataset.loader, desc=f"Train epoch {self.current_epoch+1:3d}", unit="batch", initial=self.current_batch)
        for x, y in pbar:
            if self.should_stop:
                self.save_checkpoint('last')
                exit(0)

            self.optimizer.zero_grad()
            inputs = {key: value.to(self.device) for key, value in x.items()}
            targets = {key: value.to(self.device) for key, value in y.items()}
            outputs = self.model(inputs)
            loss = self.metric(outputs, targets)
            self.current_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            self.current_batch += 1
            pbar.set_postfix({"loss": f"{self.current_loss / self.current_batch:.4f}"})
        self.history.append({'epoch': self.current_epoch + 1, 'train': self.current_loss / self.current_batch})

    def _run_test(self):
        self.model.eval()
        self.dataset.set_mode('test')
        self.dataset.set_seed(0)
        pbar = tqdm(self.dataset.loader, desc=f"Test  epoch {self.current_epoch+1:3d}", unit="batch", initial=self.current_batch)
        with torch.no_grad():
            for x, y in pbar:
                if self.should_stop:
                    self.save_checkpoint('last')
                    exit(0)

                inputs = {key: value.to(self.device) for key, value in x.items()}
                targets = {key: value.to(self.device) for key, value in y.items()}
                outputs = self.model(inputs)
                loss = self.metric(outputs, targets)

                self.current_loss += loss.item()
                self.current_batch += 1
                pbar.set_postfix({"loss": f"{self.current_loss / self.current_batch:.4f}"})
        self.history[-1]['test'] = self.current_loss / self.current_batch

    def plot_history(self):
        epochs = [item['epoch'] for item in self.history]
        train_loss = [item['train'] for item in self.history]
        test_loss = [item.get('test') for item in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, 'b-', label='Train')
        plt.scatter(epochs, train_loss, c='b', s=20)
        if any(t is not None for t in test_loss):
            plt.plot(epochs, test_loss, 'r-', label='Test')
            plt.scatter(epochs, test_loss, c='r', s=20)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.images_dir / 'history.png', dpi=150, bbox_inches='tight')
        plt.close()
