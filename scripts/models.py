import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

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

    def predict(self, X):
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
        X, y = [], []
        for X_batch, y_batch in ds.loader:
            X.append(X_batch)
            y.append(y_batch)
        X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
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
        self.params = params or {
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

    def fit(self, ds):
        X, y = [], []
        for X_batch, y_batch in ds.loader:
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
    def __init__(self, variant, model, epochs=100, lr=0.01, verbose=False, test=None):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.test = test
        self.variant = variant

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def loss_rmse(self, y_true, y_pred, lat):
        weights = torch.cos(torch.deg2rad(lat))
        return torch.sqrt(torch.mean(((y_pred - y_true) ** 2) * weights))

    def clamp(self, y_pred, climate):
        if self.variant == 'aice':
            return torch.clamp(y_pred + climate, min=0, max=1)
        elif self.variant == 'swe':
            return torch.clamp(y_pred + climate, min=0)

    def fit(self, ds):
        X, y, lat, climate = ds.load_all()
        X, y, lat, climate = X.to(self.device), y.to(self.device), lat.to(self.device), climate.to(self.device)

        if self.test is not None:
            X_test, y_test, lat_test, climate_test = self.test.load_all()
            X_test, y_test, lat_test, climate_test = X_test.to(self.device), y_test.to(self.device), lat_test.to(self.device), climate_test.to(self.device)

        train_losses, test_losses = [], []
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.clamp(self.model(X).squeeze(), climate)
            loss = self.loss_rmse(y_pred, y + climate, lat)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

            if self.test is not None:
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.clamp(self.model(X_test).squeeze().cpu(), climate_test)
                    test_losses.append(self.loss_rmse(y_pred, y_test + climate_test, lat_test))
                if self.verbose:
                    print(f'Epoch {epoch+1:3d}, train loss: {train_losses[-1]:8.4f}, test loss: {test_losses[-1]:8.4f}')
            elif self.verbose:
                print(f'Epoch {epoch+1:3d}, train loss: {train_losses[-1]:8.4f}')

        ds.set_variables(ds.variables[:-2])
        if self.test is not None:
            self.test.set_variables(self.test.variables[:-2])

        return train_losses, test_losses

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device)).squeeze().cpu()

    def save(self, filepath):
        data = {
            'model_state': self.model.state_dict(),
            'lr': self.lr,
            'epochs': self.epochs
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, filepath)

    def load(self, filepath):
        data = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(data['model_state'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=data['lr'])
        return self

class MLP(NeuralNetwork):
    def __init__(self, variant, input_size, hidden_layers=[16, 8], **kwargs):
        self.name = f'Полносвязная нейросеть'
        torch.manual_seed(42)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        net = []
        prev_size = input_size

        for size in hidden_layers:
            net.append(nn.Linear(prev_size, size))
            net.append(nn.ReLU())
            prev_size = size

        net.append(nn.Linear(prev_size, 1))
        model = nn.Sequential(*net)

        super().__init__(variant, model, **kwargs)
