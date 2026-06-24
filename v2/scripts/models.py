import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from saver import saveable

@saveable
class DummyModel(nn.Module):
    def __init__(self, target_key):
        super().__init__()
        self.target_key = target_key
        self.zero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        value = x[self.target_key].mean(dim=1) if len(x[self.target_key].shape) == 5 else x[self.target_key]
        return value + self.zero * 0

@saveable
class LinearModel(nn.Module):
    def __init__(self, target_key):
        super().__init__()
        self.target_key = target_key
        self.A = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        value = x[self.target_key].mean(dim=1) if len(x[self.target_key].shape) == 5 else x[self.target_key]
        return self.A * value + self.b

@saveable
class EnsembleEncoder(nn.Module):
    def __init__(self, num_stats, ens_size, hidden_dim=8, latent_dim=8):
        super().__init__()
        input_dim = 1
        self.ens_size = ens_size
        self.num_stats = num_stats
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.stat_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_stats)
        ])
        
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_stats, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stats)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(num_stats + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.latent_codes = nn.Parameter(torch.randn(ens_size, latent_dim) * 0.1)

    def forward(self, x):
        x = list(x.values())[0]

        B, E, T, H, W = x.shape
        N = B * T * H * W
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(N, E, -1) # (N, E, 1)
        encoded = self.encoder(x_flat)  # (N, E, hidden_dim)
        mean_encoded = encoded.mean(dim=1, keepdim=True)  # (N, 1, hidden_dim)

        stats = []
        for head in self.stat_heads:
            head_input = torch.cat([encoded, mean_encoded.expand(-1, E, -1)], dim=-1) # (N, E, hidden_dim * 2)            
            stat = head(head_input)  # (N, E, hidden_dim)
            stat = stat.mean(dim=1)  # (N, hidden_dim)
            stats.append(stat)

        combined_stats = torch.cat(stats, dim=-1)  # (N, hidden_dim * num_stats)
        encoded = self.property_predictor(combined_stats)  # (N, num_stats)
        stats_expanded = encoded.unsqueeze(1).expand(-1, E, -1) # (N, E, num_stats)
        latent = self.latent_codes.unsqueeze(0).expand(N, -1, -1) # (N, E, latent_dim)        
        decoder_input = torch.cat([stats_expanded, latent], dim=-1) # (N, E, num_stats + latent_dim)
        output = self.decoder(decoder_input) # (N, E, 1)
        output = output.reshape(B, T, H, W, E, -1).permute(0, 4, 1, 2, 3, 5).squeeze(-1) # (B, E, T, H, W)

        return output

@saveable
class StatisticEncoder(nn.Module):
    def __init__(self, ens_size, latent_dim=8):
        super().__init__()
        self.ens_size = ens_size
        self.latent_codes = nn.Parameter(torch.randn(ens_size, latent_dim) * 0.1)

    def forward(self, x):
        x = list(x.values())[0]  # (B, E, T, H, W)

        mean = x.mean(dim=1, keepdim=True)  # (B, 1, T, H, W)
        std = x.std(dim=1, keepdim=True)    # (B, 1, T, H, W)

        mean_expanded = mean.expand(-1, self.ens_size, -1, -1, -1)  # (B, E, T, H, W)
        std_expanded = std.expand(-1, self.ens_size, -1, -1, -1)    # (B, E, T, H, W)

        latent = self.latent_codes.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (1, E, 1, 1, 1, latent_dim)
        latent = latent.expand(x.size(0), -1, x.size(2), x.size(3), x.size(4), -1)  # (B, E, T, H, W, latent_dim)

        output = mean_expanded + std_expanded * latent.mean(dim=-1, keepdim=True).squeeze(-1)  # (B, E, T, H, W)
        
        return output