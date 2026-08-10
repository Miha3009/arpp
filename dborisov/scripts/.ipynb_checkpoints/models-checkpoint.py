import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.ops as ops
from tqdm.auto import tqdm
import numpy as np

def compute_loss(logits, targets, lat, alpha=0.25, gamma=2.0):
    # logits: (T, H, W)
    # targets: (T, H, W)
    # lat: (H,)
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()
    weights = weights.unsqueeze(1).expand(targets.shape[1], targets.shape[2]).unsqueeze(0)  # (T, H, W)

    focal = ops.sigmoid_focal_loss(
        logits.unsqueeze(1),  # (T, 1, H, W)
        targets.unsqueeze(1),
        alpha=alpha,
        gamma=gamma,
        reduction='none'
    )  # (T, H, W)

    accuracy = (((logits >= 0).float() == targets).float() * weights).mean().item() # (1,)
    focal = (focal * weights).mean() # (1,)

    return (focal, accuracy)

class SnowCorrectionModel(nn.Module):
    def __init__(self, forecast_variables, static_variables, embedding_dim=16):
        super().__init__()
        self.forecast_variables = forecast_variables
        self.static_variables = static_variables
        self.embedding_dim = embedding_dim

        D = embedding_dim
        D_T = 4 # time features
        D_S = len(static_variables)

        self.var_encoder = nn.ModuleDict({
            var: nn.Conv3d(1, D, kernel_size=1) for var in self.forecast_variables
        })
        self.temporal_conv1 = nn.Sequential(
            nn.Conv1d(D + D_T, D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(D + D_S, D, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(D, D, 3, padding=1),
            nn.ReLU(),
        )
        self.temporal_conv2 = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(D, D, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(D, D, 3, padding=1),
            nn.ReLU(),
        )
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(D, 1, kernel_size=1),
            nn.ReLU(),
        )
        self.temporal_conv3 = nn.Sequential(
            nn.Conv1d(D_T + 2, (D_T + 2) * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d((D_T + 2) * 2, D_T + 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(D_T + 2, 1, kernel_size=3, padding=1),
       )

    def forward(self, x):
        # x: dict {
        #   forecast_variable: (E, T, H', W'),
        #   static_variable: (H, W),
        #   'obs': (T_obs, H, W),
        #   'leadtime', 'year', 'sin_day', 'cos_day': (T,),
        #   'y': (T, H, W),
        #   'lat': (H,)
        #   'lon': (W,)
        # }
        E, T, H_, W_ = x[self.forecast_variables[0]].shape
        D = self.embedding_dim
        H, W = x['y'].shape[-2:]
        D_S = len(self.static_variables)

        x_embed = 0
        for var in self.forecast_variables:
            var_data = x[var].unsqueeze(1)  # (E, 1, T, H', W')
            var_encoded = self.var_encoder[var](var_data)     # (E, D, T, H', W')
            x_embed = x_embed + var_encoded.mean(dim=0)           # (D, T, H', W')

        x_time = torch.stack([
            x[var] for var in ['lead_time', 'year', 'sin_day', 'cos_day']
        ], dim=1)                                             # (T, D_T)
        x_embed = torch.cat([
            x_embed.permute(2, 3, 0, 1).reshape(-1, D, T),
            x_time.permute(1, 0).unsqueeze(0).expand(H_ * W_, -1, -1)
        ], dim=1)  # (H' * W', D + D_T, T)
        x_embed = self.temporal_conv1(x_embed)        # (H' * W', D, T)

        x_embed = x_embed.reshape(H_, W_, D, T).permute(2, 3, 0, 1)  # (D, T, H', W')
        x_embed = nn.functional.interpolate(
                x_embed.reshape(D * T, 1, H_, W_),
                size=(H, W),
                mode='bilinear',
                align_corners=False
        ).reshape(D, T, H, W)  # (D, T, H, W)

        x_static = torch.stack(
            [x[var] for var in self.static_variables]
        ) # (D_S, H, W)
        x_static = x_static.unsqueeze(0).expand(T, -1, -1, -1) # (T, D_S, H, W)
        x_embed = x_embed.permute(1, 0, 2, 3) # (T, D, H, W)
        x_embed = torch.cat([x_embed, x_static], dim=1) # (T, D + D_S, H, W)
        x_embed = self.spatial_conv1(x_embed) # (T, D, H, W)

        x_embed = x_embed.permute(2, 3, 1, 0).reshape(-1, D, T)
        x_embed = self.temporal_conv2(x_embed) # (H * W, D, T)

        x_embed = x_embed.reshape(H, W, D, T).permute(3, 2, 0, 1)  # (T, D, H, W)
        x_embed = self.spatial_conv2(x_embed) # (T, D, H, W)
        x_embed = self.channel_reducer(x_embed) # (T, 1, H, W)

        x_obs = x['obs'].mean(dim=0) # (H, W)
        x_obs = x_obs.reshape(H * W, 1, 1).expand(-1, -1, T) # (H * W, 1, T)
        x_embed = x_embed.permute(2, 3, 1, 0).reshape(H * W, 1, T) # (H * W, 1, T)
        x_time = x_time.permute(1, 0).unsqueeze(0).expand(H * W, -1, -1) # (H * W, D_T, T)
        x_embed = torch.cat([x_embed, x_obs, x_time], dim=1)  # (H * W, D_T + 2, T)
        x_embed = self.temporal_conv3(x_embed) # (H * W, 1, T)

        logits = x_embed.squeeze(1).reshape(H, W, T).permute(2, 0, 1) # (T, H, W)
        return logits

def train_model(ds, epochs=10, embedding_dim=16, alpha=0.25, gamma=2.0, lr=1e-3):
    model = SnowCorrectionModel(
        forecast_variables=ds.forecast_variables,
        static_variables=ds.static_variables,
        embedding_dim=embedding_dim
    ).to(ds.device)

    #for name, param in model.named_parameters():
        #if param.requires_grad:
           # print(f"{name}: {param.numel():,} параметров")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nВсего: {total:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        totla_basic_loss = 0.0
        total_acc = 0.0
        total_basic_acc = 0.0
        total_class = 0.0
        n_batches = 0

        with tqdm(ds.loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for data in pbar:
                for k in data:
                    data[k] = data[k].to(device=ds.device)
                logits = model(data)

                loss, acc = compute_loss(logits, data['y'], data['lat'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += acc
                total_class += 1 - torch.mean(data['y'])
                n_batches += 1

                #basic_logits = torch.mean(torch.where(data['SS'] > 5, 10.0, -10.0), axis=0)
                #basic_loss, basic_acc = model.compute_loss(basic_logits, data['y'], data['y_lat']).item()
                #total_basic_acc += basic_acc
                pbar.set_postfix({'loss': f'{total_loss / n_batches:.5f}',
                 #                 'basic_loss': f'{total_basic_loss / n_batches:.5f}',
                                  'acc': f'{total_acc / n_batches:.4f}',
                                  'class': f'{total_class / n_batches:.4f}'})
                  #                'basic_acc': f'basic_loss: {total_basic_acc / n_batches:.4f}'})

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}: train_loss = {avg_loss:.6f}")
    
    return model
