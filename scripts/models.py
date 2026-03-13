import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
import time
import os

class ClimateModel:
    def __init__(self):
        self.name = 'Климат'

    def fit(self, ds):
        return

    def predict(self, X):
        return torch.zeros(X.shape[0])
    
    def save(self, filepath):
        return
    
    def load(self, filepath):
        return self

class BaseModel:
    def __init__(self):
        self.name = 'Несмещенный прогноз'

    def fit(self, ds): 
        if 'swe' in ds.variables:
            var_name = 'swe'
        elif 'aice' in ds.variables:
            var_name = 'aice'
        self.idx = ds.variables.index(var_name)
        self.std = ds.std[var_name] if ds.normed else 1
        self.loader_type = ds.loader_type

    def predict(self, X):
        if self.loader_type == 'sequence':
            return X[-1, :, self.idx] * self.std        
        elif self.loader_type == 'map':
            return X[:, :, :, self.idx] * self.std
        elif self.loader_type == 'sequence_map':
            return X[-1, :, :, :, self.idx] * self.std
        return X[:, self.idx] * self.std
    
    def save(self, filepath):
        return
    
    def load(self, filepath):
        return self

class LinearRegression:
    def __init__(self, variables=None):
        self.name = 'Линейная регрессия'
        self.weights = None
        self.bias = None
        self.variables = variables

    def fit(self, ds):
        X, y, _ = ds.load_all()
        self.__fit__(X, y)

    def __fit__(self, X, y):
        ones = torch.ones(X.shape[0], 1)
        X = torch.cat([ones, X], dim=1)
        self.weights = torch.linalg.solve(X.T @ X, X.T @ y)
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Модель не обучена")
        return X @ self.weights + self.bias

    def save(self, filepath):
        data = {
            'weights': self.weights,
            'bias': self.bias,
            'variables': self.variables
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, filepath)

    def load(self, filepath):
        data = torch.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']
        self.variables = data['variables']
        return self

class BoostingModel:
    def __init__(self, variables, params={}):
        self.name = 'LightGBM'
        self.model = None
        self.variables = variables
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'max_depth': -1,
            'lambda_l1': 0.01,
            'num_boost_round': 150,
            'learning_rate': 0.06,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'verbose': -1,
            'seed': 42
        }
        self.params.update(params)

    def fit(self, ds):
        X, y = [], []
        for X_batch, y_batch, _ in ds.loader:
            X.append(X_batch.numpy())
            y.append(y_batch.numpy())
        X, y = np.vstack(X), np.concatenate(y)

        train_data = lgb.Dataset(X, label=y, feature_name=self.variables)
        self.model = lgb.train(self.params, train_data)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена")

        X_df = pd.DataFrame(X.numpy(), columns=self.variables)
        pred_np = self.model.predict(X_df)
        return torch.from_numpy(pred_np).float()

    def save(self, filepath):
        data = {
                'model': self.model,
                'params': self.params,
                'variables': self.variables
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.model = data['model']
        self.params = data['params']
        self.variables = data['variables']
        return self

class NeuralNetwork:
    def __init__(self, variant, model_class, model_args, name='Нейросеть', epochs=100, lr=0.01, verbose=True, test=None,
                 acc_steps=1, filepath='tmp', history=10, loss_type='rmse'):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.test = test
        self.variant = variant
        self.model_class = model_class
        self.model_args = model_args
        self.acc_steps = acc_steps
        self.name = name
        self.filepath = filepath
        self.history = history
        self.loss_type = loss_type

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_class(**model_args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.current_epochs = 0
        self.train_losses = []
        self.test_losses = []

    def loss_rmse(self, y_pred, y_true, lat):
        if y_true.dim() > y_pred.dim():
            y_true = y_true[-1, ...]
            lat = lat[-1, ...]

        y_true, y_pred, lat = y_true.flatten(), y_pred.flatten(), lat.flatten()
        weights = torch.cos(torch.deg2rad(lat))

        return torch.sqrt(torch.mean(((y_pred - y_true) ** 2) * weights))

    def loss_acc(self, y_pred, y_true, lat):
        #if y_true.dim() > y_pred.dim():
            #y_true = y_true[-1, ...]
            #lat = lat[-1, ...]

        y_true, y_pred, lat = y_true.flatten(), y_pred.flatten(), lat.flatten()
        weights = torch.cos(torch.deg2rad(lat))

        cov = torch.sum(weights * y_true * y_pred)
        std_true = torch.sqrt(torch.sum(weights * y_true ** 2))
        std_pred = torch.sqrt(torch.sum(weights * y_pred ** 2))
        return -cov / (std_true * std_pred + 1e-6)

    def fit_test(self, ds, dsTest):
        self.test = dsTest
        return self.fit(ds)

    def fit(self, ds):
        for epoch in range(self.current_epochs, self.epochs):
            self.model.train()
            epoch_start = time.time()
            epoch_loss, sample_count = 0, 0
            i = 0
            for X_batch, y_batch, lat_batch in ds.loader:
                i += 1
                X_batch, y_batch, lat_batch = X_batch.to(self.device), y_batch.to(self.device), lat_batch.to(self.device)
                for step in range(self.acc_steps):
                    self.optimizer.zero_grad()
                    y_pred = self.model(X_batch).squeeze()
                    if self.loss_type == 'rmse':
                        loss = self.loss_rmse(y_pred, y_batch.squeeze(), lat_batch.squeeze())
                    elif self.loss_type == 'acc':
                        loss = self.loss_acc(y_pred, y_batch.squeeze(), lat_batch.squeeze())
                    loss.backward()
                    self.optimizer.step()
                    n = len(y_batch.squeeze())
                    epoch_loss += loss.item() * n
                    sample_count += n
                if self.verbose and epoch == 0 and i % 10 == 0:
                    print(f'Batch {i:4d}, {epoch_loss / sample_count:8.4f}')

            self.train_losses.append(epoch_loss / sample_count)

            if self.test is not None:
                self.model.eval()
                with torch.no_grad():
                    y_true, y_pred, lat = [], [], []
                    for X_batch, y_batch, lat_batch in self.test.loader:
                        X_batch, y_batch, lat_batch = X_batch.to(self.device), y_batch.to(self.device), lat_batch.to(self.device)
                        y_pred.append(self.model(X_batch).squeeze())
                        if ds.loader_type == 'sequence' or ds.loader_type == 'sequence_map':
                            y_true.append(y_batch[-1, :])
                            lat.append(lat_batch[-1, :])
                        elif ds.loader_type == 'point' or ds.loader_type == 'map':
                            y_true.append(y_batch)
                            lat.append(lat_batch)
                        if len(y_pred[-1].shape) < len(y_true[-1].shape):
                            y_pred[-1] = y_pred[-1].unsqueeze(0)
                    y_true, y_pred, lat = torch.cat(y_true), torch.cat(y_pred), torch.cat(lat)
                    self.test_losses.append(-self.loss_acc(y_pred, y_true, lat).item())
                    #self.test_losses.append(self.loss_rmse(y_pred, y_true, lat).item())
                if self.verbose and (self.current_epochs < 10 or self.epochs < 300 or self.current_epochs % 10 == 9):
                    print(f'Epoch {epoch+1:3d}, train loss: {self.train_losses[-1]:8.4f}, test loss: ' +
                          f'{self.test_losses[-1]:8.4f}, time: {(time.time() - epoch_start):.2f}s')
            elif self.verbose:
                print(f'Epoch {epoch+1:3d}, train loss: {self.train_losses[-1]:8.4f}')

            self.current_epochs += 1
            if not self.filepath is None:
                self.save(self.filepath)
                if self.history is not None and self.current_epochs % self.history == 0:
                    self.save(self.filepath + f'_hist{self.current_epochs:04d}')

        return self.train_losses, self.test_losses

    def fit_simple(self, X, y, lat, X_test, y_test, lat_test):
        X, y, lat = X.to(self.device), y.to(self.device), lat.to(self.device)
        X_test, y_test, lat_test = X_test.to(self.device), y_test.to(self.device), lat_test.to(self.device)
        for epoch in range(self.current_epochs, self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(X).squeeze()
            loss = self.loss_acc(y_pred, y.squeeze(), lat.squeeze())
            loss.backward()
            self.optimizer.step()
            self.train_losses.append(-loss.item())

            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_test).squeeze()
                self.test_losses.append(-self.loss_acc(y_pred, y_test.squeeze(), lat_test.squeeze()).item())

        return self.train_losses, self.test_losses


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device)).squeeze().cpu()

    def save(self, filepath):
        data = {
            'model_state': self.model.state_dict(),
            'model_class': self.model_class,
            'model_args': self.model_args,
            'variant': self.variant,
            'name': self.name,
            'lr': self.lr,
            'epochs': self.epochs,
            'current_epochs': self.current_epochs,
            'verbose': self.verbose,
            'acc_steps': self.acc_steps,
            'filepath': self.filepath,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'history': self.history,
            'loss_type': self.loss_type,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, filepath + '.model')

def load_nn(filepath, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(filepath + '.model', map_location=device, weights_only=False)

    nn_instance = NeuralNetwork(
        variant=data['variant'],
        model_class=data['model_class'],
        model_args=data['model_args'],
        name=data['name'],
        lr=data['lr'],
        epochs=data['epochs'],
        verbose=data['verbose'],
        acc_steps=data['acc_steps'],
        filepath=data['filepath'],
        history=data['history'],
        loss_type=data['loss_type'],
    )
    nn_instance.current_epochs = data['current_epochs']
    nn_instance.train_losses = data['train_losses']
    nn_instance.test_losses = data['test_losses']
    nn_instance.model.load_state_dict(data['model_state'])
    return nn_instance

def create_nn(variant, model_class, model_args, filepath=None, test=None, seed=42, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    if not filepath is None and os.path.exists(filepath + '.model'):
        nn_instance = load_nn(filepath)
        nn_instance.test = test
        return nn_instance
    return NeuralNetwork(variant, model_class, model_args, filepath=filepath, test=test, **kwargs)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers=[16, 8], dropout=0):
        super().__init__()
        net = []
        prev_size = input_size
        for size in hidden_layers:
            net.append(nn.Linear(prev_size, size))
            net.append(nn.ReLU())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            prev_size = size
        net.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=4, bidirectional=True):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, 
            hidden_size, 
            batch_first=False,
            bidirectional=bidirectional
        )
        out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(out_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_out = out[-1]
        return self.fc(last_out)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=4, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[-1, :, :])
    
class TCNModel(nn.Module):
    def __init__(self, input_size, channels=[8, 16], kernel_size=3, seqlen=None):
        super().__init__()
        self.seqlen = seqlen
        self.convs = nn.ModuleList()
        last_size = input_size
        for channel in channels:
            self.convs.append(nn.Conv1d(last_size, channel, kernel_size, padding=kernel_size // 2))
            last_size = channel
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(last_size, 1)

    def forward(self, x):
        # [seq_len, batch, features]
        x = x.permute(1, 2, 0)  # [batch, features, seq_len]
        if self.seqlen is not None:
            x = x[:, :, -self.seqlen:]
        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](x))
        x = self.pool(x).squeeze(-1)  # [batch, channels]
        return self.fc(x)

class HybridTCNModel(nn.Module):
    def __init__(self, input_size, conv_channels=[16, 32], mlp_layers=[64, 32], kernel_size=3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        last_c = input_size
        for c in conv_channels:
            self.convs.append(nn.Conv1d(last_c, c, kernel_size, padding=kernel_size // 2))
            last_c = c

        self.mlp = nn.ModuleList()
        last_m = conv_channels[-1]
        for m in mlp_layers:
            self.mlp.append(nn.Linear(last_m, m))
            last_m = m
            
        self.fc = nn.Linear(last_m, 1)

    def forward(self, x):
        # [seq_len, batch, features] -> [batch, features, seq_len]
        x = x.permute(1, 2, 0)
    
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x[:, :, -1] 
        for layer in self.mlp:
            x = F.relu(layer(x))
        return self.fc(x)

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetModel(nn.Module):
    def __init__(self, input_channels, base_channels=32, batch_norm=False):
        super().__init__()
        self.enc1 = ConvBlock2D(input_channels, base_channels)                                  # 200x1440xb
        self.pool1 = nn.MaxPool2d((2, 2))                                                       # 100x720 xb
        self.enc2 = ConvBlock2D(base_channels, base_channels*2)                                 # 100x720 x2b
        self.pool2 = nn.MaxPool2d((2, 2))                                                       # 50 x360 x2b
        self.enc3 = ConvBlock2D(base_channels*2, base_channels*4)                               # 50 x360 x4b
        self.pool3 = nn.MaxPool2d((2, 2))                                                       # 25 x180 x4b
        self.bottleneck = ConvBlock2D(base_channels*4, base_channels*8)                         # 25 x180 x8b
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, (2, 2), stride=(2, 2))  # 50 x360 x4b
        self.dec3 = ConvBlock2D(base_channels*8, base_channels*4)                               # 50 x360 x4b
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, (2, 2), stride=(2, 2))  # 100x720 x2b
        self.dec2 = ConvBlock2D(base_channels*4, base_channels*2)                               # 100x720 x2b
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, (2, 2), stride=(2, 2))    # 200x1440xb
        self.dec1 = ConvBlock2D(base_channels*2, base_channels)                                 # 200x1440xb
        self.final_conv = nn.Conv2d(base_channels, 1, 1)                                        # 200x1440x1
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [batch, features, 201, 1440]
        h, w = x.shape[2], x.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        e1 = self.enc1(x_padded)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final_conv(d1)
    
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out.squeeze(1)

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet3DModel(nn.Module):
    def __init__(self, input_channels, base_channels=16):
        super().__init__()
        self.enc1 = ConvBlock3D(input_channels, base_channels)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2 = ConvBlock3D(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.enc3 = ConvBlock3D(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.bottleneck = ConvBlock3D(base_channels*4, base_channels*8)
        self.up3 = nn.ConvTranspose3d(base_channels*8, base_channels*4, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = ConvBlock3D(base_channels*8, base_channels*4)
        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlock3D(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock3D(base_channels*2, base_channels)
        self.final_conv = nn.Conv3d(base_channels, 1, 1)

    def forward(self, x):
        x = x.permute(1, 4, 0, 2, 3)         # [batch, features, seq, 201, 1440]
        x_cropped = x[:, :, :, 1:, :]
        e1 = self.enc1(x_cropped)            # [batch,  b, seq, 200, 1440]
        e2 = self.enc2(self.pool1(e1))       # [batch, 2b, seq, 100,  720]
        e3 = self.enc3(self.pool2(e2))       # [batch, 4b, seq,  50,  360]
        b = self.bottleneck(self.pool3(e3))  # [batch, 8b, seq,  25,  180]
        d3 = self.up3(b)                     # [batch, 4b, seq,  50,  360]
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)                    # [batch, 2b, seq, 100,  720]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)                    # [batch,  b, seq, 200, 1440]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out_cropped = self.final_conv(d1)    # [batch,  1, seq, 200, 1440]
        out_full = torch.zeros(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device)
        out_full[:, :, :, 1:, :] = out_cropped
        r = out_full[:, :, -1, :, :].squeeze()

        return r            # [batch, seq, 201, 1440]

class CNNModel(nn.Module):
    def __init__(self, input_channels, base_channels=32, layers_count=3, dropout=0, batch_norm=False):
        super().__init__()
        
        net = []
        prev_channels = input_channels
        
        for i in range(layers_count):
            channels = base_channels
            net.append(nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1))
            if batch_norm:
                net.append(nn.BatchNorm2d(channels))
            net.append(nn.ReLU(inplace=True))
            if dropout > 0:
                net.append(nn.Dropout2d(dropout))
            prev_channels = channels

        net.append(nn.Conv2d(prev_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [batch, features, 201, 1440]
        x = self.net(x)
        return x.squeeze(1)

class CNN3DModel(nn.Module):
    def __init__(self, input_channels, base_channels=32, layers_count=3, dropout=0, batch_norm=False):
        super().__init__()
        
        net = []
        prev_channels = input_channels
        
        for i in range(layers_count):
            channels = base_channels
            net.append(nn.Conv3d(prev_channels, channels, kernel_size=3, padding=1))
            if batch_norm:
                net.append(nn.BatchNorm3d(channels))
            net.append(nn.ReLU(inplace=True))
            if dropout > 0:
                net.append(nn.Dropout3d(dropout))
            prev_channels = channels

        net.append(nn.Conv3d(prev_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.net(x)
        return x.squeeze(1)[-1, :, :, :]