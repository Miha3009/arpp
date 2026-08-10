import torch
import torch.nn as nn
from pathlib import Path
import signal
import pandas as pd
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import json
from collections import defaultdict
from datetime import datetime, timedelta
import xarray as xr

from saver import load_from_config

class Experiment:
    def __init__(self, name, dataset=None, model=None, metric=None, optimizer=None, modes=['train', 'test'],
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

            self.modes = modes
            self.current_epoch = 0
            self.current_batch = 0
            self.current_loss = 0
            self.current_mode = modes[0]
            self.history = []
            self._save_config()

        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print('\nОбучение прерывается...')
        self.should_stop = True

    def _save_config(self):
        self.dataset.save_config(self.exp_dir / 'dataset.json')
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
            'modes': self.modes,
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

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.optimizer.setup(self.model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.modes = checkpoint.get('modes', ['train', 'test'])
        self.current_mode = checkpoint['current_mode']
        self.current_epoch = checkpoint['current_epoch']
        self.current_batch = checkpoint['current_batch']
        self.current_loss = checkpoint['current_loss']
        self.history = checkpoint['history']

    def run(self, epochs, save_every_n_epochs=5):
        while self.current_epoch < epochs:
            if self.current_mode == self.modes[0] and (len(self.history) == 0 or self.history[-1]['epoch'] != self.current_epoch + 1):
                self.history.append({'epoch': self.current_epoch + 1})
            self.dataset.loader.set_skip(self.current_batch)
            if 'train' in self.current_mode:
                self._run_train()
            else:
                self._run_test()
                self.update_history()
                self.plot_history()
                self.current_epoch += 1
            self.current_mode = self.modes[(self.modes.index(self.current_mode) + 1) % len(self.modes)]
            self.current_batch = 0
            self.current_loss = 0
            if self.current_mode == self.modes[0] and self.current_epoch % save_every_n_epochs == 0 and self.current_epoch < epochs:
                self.save_checkpoint(f'e{self.current_epoch:03d}')
        self.save_checkpoint(f'last')

    def _run_train(self):
        self.model.train()
        self.dataset.set_mode(self.current_mode)
        self.dataset.set_seed(self.current_epoch)
        if hasattr(self.metric, 'set_mode'):
            self.metric.set_mode('train')
        pbar = tqdm(self.dataset.loader, desc=f"{self.current_mode} epoch {self.current_epoch+1:3d}", unit="batch", initial=self.current_batch)
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
        self.history[-1][self.current_mode] = self.current_loss / self.current_batch

    def _run_test(self):
        self.model.eval()
        self.dataset.set_mode(self.current_mode)
        self.dataset.set_seed(0)
        if hasattr(self.metric, 'set_mode'):
            self.metric.set_mode('test')
        pbar = tqdm(self.dataset.loader, desc=f"{self.current_mode}  epoch {self.current_epoch+1:3d}", unit="batch", initial=self.current_batch)
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
        self.history[-1][self.current_mode] = self.current_loss / self.current_batch

    def plot_history(self):
        history = pd.DataFrame(self.history)
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

    def plot_history(self):
        df = pd.DataFrame(self.history)

        columns = {}
        for col in df.columns:
            if col == 'epoch':
                continue
            if '#' in col:
                mode, metric = col.split('#', 1)
                columns[col] = (mode, metric)
            else:
                columns[col] = (col, 'loss')

        metric_groups = defaultdict(list)
        for col, (mode, metric) in columns.items():
            metric_groups[metric].append((mode, col))
    
        colors = ['b', 'r', 'g', 'c', 'm']
        mode_colors = {}
        color_idx = 0
        for metric, items in metric_groups.items():
            all_vals = []
            x = df['epoch']
            plt.figure(figsize=(8, 5))
            for mode, col in items:
                if mode not in mode_colors:
                    mode_colors[mode] = colors[color_idx % len(colors)]
                    color_idx += 1

                color = mode_colors[mode]
                values = df[col]
                all_vals.extend(values)

                plt.plot(x, values, f'{color}-', label=mode)
                plt.scatter(x, values, c=color, s=20)
        
            plt.xlabel('Epoch')
            x = x.iloc[-20:]
            plt.xticks(x.astype(int))
            plt.xlim([min(x)-0.5, max(x)+0.5])
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(self.images_dir / f'history_{metric}.png', dpi=150, bbox_inches='tight')
            plt.close()

    def update_history(self):
        flattened = {}

        def flatten(d, parent_key='', sep='#'):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten(v, new_key, sep)
                elif isinstance(v, torch.Tensor):
                    flattened[new_key] = v.item()
                else:
                    flattened[new_key] = v

        flatten(self.history[-1])
        self.history[-1] = flattened

    def run_full_test(self, mode, target_field):
        self.dataset.set_full(mode)
        pbar = tqdm(self.dataset.loader, desc=mode, unit="batch")
        B = len(self.dataset.xyt)
        W = self.dataset.target_scale['xSize']
        H = self.dataset.target_scale['ySize']
        T = self.dataset.target_scale['tSize']
        result = {
            'output': torch.full((B, T, H, W), float('nan'), dtype=torch.float32),
            'target': torch.full((B, T, H, W), float('nan'), dtype=torch.float32),
            'lat': torch.full((B, H), float('nan'), dtype=torch.float32),
            'lon': torch.full((B, W), float('nan'), dtype=torch.float32),
            'day': torch.full((B, T), float('nan'), dtype=torch.float32),
            'year': torch.full((B, T), float('nan'), dtype=torch.float32),
            'lead_time': torch.full((B, T), float('nan'), dtype=torch.float32),
        }
        i = 0
        with torch.no_grad():
            for x, y in pbar:
                inputs = {key: value.to(self.device) for key, value in x.items()}
                targets = {key: value.to(self.device) for key, value in y.items()}
                outputs = self.model(inputs)
                for j in range(outputs.shape[0]):
                    result['output'][i+j, ...] = outputs[j, ...]
                    result['target'][i+j, ...] = targets[target_field][j, ...]
                    for rfield, field in [('target', target_field), ('lat', 'lat'), ('lon', 'lon'), ('day', 'day'),
                                  ('year', 'year'), ('lead_time', 'inm_lead_time')]:
                        result[rfield][i+j, ...] = targets[field][j, ...]
                i += outputs.shape[0]
        i = 0
        mode = self.dataset.modes[mode]
        T2 = mode['lead_time_range'][1] - mode['lead_time_range'][0] + T
        H2 = mode['y_max'] - mode['y_min']
        W2 = mode['x_max'] - mode['x_min']
        output_dir = self.exp_dir / 'result'
        output_dir.mkdir(parents=True, exist_ok=True)
        for t, B in tqdm(self.dataset.t.items(), desc='save', unit='file'):
            current = {
                'output': torch.full((T2, H2, W2), float('nan'), dtype=torch.float32),
                'target': torch.full((T2, H2, W2), float('nan'), dtype=torch.float32),
                'lat': torch.full((H2,), float('nan'), dtype=torch.float32),
                'lon': torch.full((W2,), float('nan'), dtype=torch.float32),
                'day': torch.full((T2,), float('nan'), dtype=torch.float32),
                'year': torch.full((T2,), float('nan'), dtype=torch.float32),
                'lead_time': torch.full((T2,), float('nan'), dtype=torch.float32),
            }
            for j in range(i + B - 1, i - 1, -1):
                x, y, l = self.dataset.xyl[j]
                x1, x2 = x - W // 2, x + W // 2
                y1, y2 = y - H // 2, y + H // 2
                l1, l2 = l - T, l
                crop_x, crop_y = max(x2 - W2, 0), max(y2 - H2, 0)
                for key in ['output', 'target']:
                    current[key][l1:l2, y1:y2-crop_y, x1:x2-crop_x] = result[key][j, :, :H-crop_y, :W-crop_x]
                current['lat'][y1:y2-crop_y] = result['lat'][j, :H-crop_y]
                current['lon'][x1:x2-crop_x] = result['lon'][j, :W-crop_x]
                for key in ['day', 'year', 'lead_time']:
                    current[key][l1:l2] = result[key][j, :]
            ds = xr.Dataset({
                'output': (['lead_time', 'lat', 'lon'], current['output'].numpy()),
                'target': (['lead_time', 'lat', 'lon'], current['target'].numpy()),
                'day': (['lead_time'], current['day'].numpy()),
                'year': (['lead_time'], current['year'].numpy()),
            }, coords={
                'lat': current['lat'].numpy(),
                'lon': current['lon'].numpy(),
                'lead_time': current['lead_time'].numpy(),
            })
            t_str = (datetime(1970, 1, 1) + timedelta(days=int(t))).strftime('%Y%m')
            ds.to_netcdf(output_dir / f'{t_str}.nc', engine='h5netcdf', encoding={
                'output': {'zlib': True, 'complevel': 4},
                'target': {'zlib': True, 'complevel': 4},
            })
            i += B

