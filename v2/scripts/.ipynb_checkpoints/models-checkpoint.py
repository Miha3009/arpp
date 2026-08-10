import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from saver import saveable, load_from_config_content
from collections import defaultdict
from scales import normalize

@saveable
class DummyModel(nn.Module):
    def __init__(self, target_key, size=None):
        super().__init__()
        self.target_key = target_key
        self.size = size
        self.zero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        value = x[self.target_key].mean(dim=1) if len(x[self.target_key].shape) == 5 else x[self.target_key]
        if self.size is not None:
            value = F.interpolate(value, self.size, mode='bilinear', align_corners=False)
        return value + self.zero * 0

@saveable
class LinearModel(nn.Module):
    def __init__(self, target_key, size=None):
        super().__init__()
        self.target_key = target_key
        self.size = size
        self.A = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        value = x[self.target_key].mean(dim=1) if len(x[self.target_key].shape) == 5 else x[self.target_key]
        if self.size is not None:
            value = F.interpolate(value, self.size, mode='bilinear', align_corners=False)
        return self.A * value + self.b

@saveable
class EnsembleEncoder(nn.Module):
    def __init__(self, num_stats, ens_size, scale, hidden_dim=8, latent_dim=8):
        super().__init__()
        input_dim = 1
        self.ens_size = ens_size
        self.num_stats = num_stats
        self.hidden_dim = hidden_dim
        self.scale = scale

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

    def encode(self, x):
        B, E, T, H, W = x.shape
        N = B * T * H * W
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(N, E, -1) # (N, E, 1)
        x_flat /= self.scale
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
        return encoded

    def decode(self, encoded, shape):
        B, E, T, H, W = shape
        N = B * T * H * W

        stats_expanded = encoded.unsqueeze(1).expand(-1, E, -1) # (N, E, num_stats)
        latent = self.latent_codes[:E].unsqueeze(0).expand(N, -1, -1) # (N, E, latent_dim)
        decoder_input = torch.cat([stats_expanded, latent], dim=-1) # (N, E, num_stats + latent_dim)
        output = self.decoder(decoder_input) # (N, E, 1)
        output = output.reshape(B, T, H, W, E, -1).permute(0, 4, 1, 2, 3, 5).squeeze(-1) # (B, E, T, H, W)
        output *= self.scale
        return output

    def forward(self, x):
        x = list(x.values())[0]
        return self.decode(self.encode(x), x.shape)

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

class MultiscaleNet(nn.Module):
    def __init__(self, variable_encoders):
        super().__init__()
        self._init_encoders(variable_encoders)
        self._freeze_encoders()
    
    def _init_encoders(self, variable_encoders):
        self.variable_encoders_configs = {}
        self.variable_encoders = nn.ModuleDict()
        for v, p in variable_encoders.items():
            if not os.path.exists(p):
                self.variable_encoders[v] = None
                continue

            with open(f'{p}/model.json', 'r') as f:
                encoder_config = json.load(f)
            self.variable_encoders_configs[v] = encoder_config
            encoder = load_from_config_content(encoder_config)
            checkpoint = torch.load(f'{p}/checkpoints/last.pt')
            encoder.load_state_dict(checkpoint['model_state_dict'])
            self.variable_encoders[v] = encoder

    def _freeze_encoders(self):
        for encoder in self.variable_encoders.values():
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False
    
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        configs_json = json.dumps(self.variable_encoders_configs)
        state['variable_encoders_configs'] = torch.tensor([ord(c) for c in configs_json], dtype=torch.uint8)
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        configs_tensor = state_dict.pop('variable_encoders_configs')
        self.variable_encoders_configs = json.loads(''.join(chr(int(x)) for x in configs_tensor))
        for v, config in self.variable_encoders_configs.items():
            encoder = load_from_config_content(config)
            self.variable_encoders[v] = encoder
        self._freeze_encoders()

        super().load_state_dict(state_dict, strict)

    def _get_scale_varname(self, v):
        if v.startswith('inm_'):
            scale = v.split('_')[-1]
            varname = v[4:-len(scale)-1]
            scale = "inm_" + scale
        else:
            scale = v.split('_')[-1]
            varname = v[:-len(scale)-1]
        return scale, varname
    
    def forward(self, x):
        result = defaultdict(dict)
        for v in x.keys():
            scale, varname = self._get_scale_varname(v)
            if 'x' not in result[scale]:
                result[scale]['x'] = []
            if len(x[v].shape) == 2:
                result[scale][varname] = normalize(x[v], varname)
                continue

            if 'inm' not in scale:
                result[scale]['x'].append(normalize(x[v].unsqueeze(1), varname))
                continue

            if varname == 'snow_cover':
                result[scale]['x'].append(x[v].mean(dim=1, keepdim=True))
            elif varname in self.variable_encoders and self.variable_encoders[varname] is not None:
                B, _, T, H, W = x[v].shape
                encoded = self.variable_encoders[varname].encode(normalize(x[v], varname)).reshape(B, -1, T, H, W)
                result[scale]['x'].append(encoded)
            else:
                result[scale]['x'].append(normalize(x[v].mean(dim=1, keepdim=True), varname))

        for scale in result.keys():
            result[scale]['x'] = torch.cat(result[scale]['x'], dim=1)

        return result

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, padding=1, pool_size=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv1d(in_channels if i==0 else out_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool1d(pool_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, padding=1, pool_size=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(pool_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UpscaleBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv1d(in_channels if i==0 else out_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x

class UpscaleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)
        self.in_channels = in_channels

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x

class ScaleEncoder(nn.Module):
    def __init__(self, in_channels, blocks, t_blocks, temporal_channels=0, save_skip=False, n_convs_per_block=2):
        super().__init__()
        self.spatial_blocks = nn.ModuleList()
        for i in range(blocks):
            self.spatial_blocks.append(ConvBlock2d(in_channels, in_channels*2, n_convs_per_block))
            in_channels *= 2

        in_channels += temporal_channels
        self.temporal_blocks = nn.ModuleList()
        for i in range(t_blocks):
            self.temporal_blocks.append(ConvBlock1d(in_channels, in_channels, n_convs_per_block))
        self.save_skip = save_skip

    def forward(self, x, temporal):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        skip = []
        if self.save_skip:
            skip.append(x)
        for block in self.spatial_blocks:
            x = block(x)
            if self.save_skip:
                skip.append(x)
        skip = skip[:-1]
    
        _, C2, H2, W2 = x.shape
        x = x.reshape(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4)
        temporal = temporal.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H2, W2)
        x = torch.cat([x, temporal], dim=1)

        if len(self.temporal_blocks) == 0:
            return x, skip

        C3 = x.shape[1]
        x = x.permute(0, 3, 4, 1, 2).reshape(B*H2*W2, C3, T)
        if self.save_skip:
            skip.append(x)
        for block in self.temporal_blocks:
            x = block(x)
            if self.save_skip:
                skip.append(x)
        skip = skip[:-1]
        T2 = x.shape[-1]
        x = x.reshape(B, H2, W2, C3, T2).permute(0, 3, 4, 1, 2)
        return x, skip

class TokenEncoder(nn.Module):
    def __init__(self, in_channels, token_size, n_layers=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_channels if i == 0 else token_size, token_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(token_size, token_size))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class TokenUpscaler(nn.Module):
    def __init__(self, token_size, blocks, t_blocks, skip_size):
        super().__init__()
        self.temporal_blocks = nn.ModuleList([
            UpscaleBlock1d(token_size + skip_size[i], token_size, n_convs=2, scale_factor=2)
            for i in range(t_blocks)
        ])
        self.spatial_blocks = nn.ModuleList([
            UpscaleBlock2d(token_size // 2**i + skip_size[i + t_blocks], token_size // 2**(i+1), n_convs=2, scale_factor=2)
            for i in range(blocks)
        ])
    
    def forward(self, x, skip):
        B, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, C, T)

        for i, block in enumerate(self.temporal_blocks):
            x = block(x, skip[i])

        _, C2, T2 = x.shape
        x = x.reshape(B, H, W, C2, T2).permute(0, 4, 3, 1, 2).reshape(B*T2, C2, H, W)
        for i, block in enumerate(self.spatial_blocks):
            x = block(x, skip[i + len(self.temporal_blocks)])
        _, C3, H2, W2 = x.shape

        return x

@saveable
class SnowCoverNetV1(MultiscaleNet):
    def __init__(self, variable_encoders, scales, token_size):
        super().__init__(variable_encoders)
        self.scale_encoders = nn.ModuleDict()
        self.token_encoders = nn.ModuleDict()
        self.scales = scales
        self.token_size = token_size
        for s in scales:
            scale = scales[s]
            self.scale_encoders[s] = ScaleEncoder(in_channels=scale['channels'],
                                                  blocks=scale['blocks'],
                                                  t_blocks=scale['t_blocks'],
                                                  temporal_channels=len(scale['temporal']),
                                                  save_skip=scale.get('skip', False))
            token_input = scale['channels'] * 2 ** scale['blocks'] + len(scale['temporal'])
            self.token_encoders[s] = TokenEncoder(token_input, token_size)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=token_size,
                nhead=8,
                dim_feedforward=token_size*4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        skip_size = []
        for i in range(scales['inm_regional']['t_blocks']):
            skip_size.append(scales['inm_regional']['channels'] * 2 ** scales['inm_regional']['blocks'] + len(scales['inm_regional']['temporal']))
        for i in range(1, scales['inm_regional']['blocks'] + 1):
            skip_size.append(scales['inm_regional']['channels'] * 2 ** (scales['inm_regional']['blocks'] - i))
        for i in range(2):
            skip_size.append(scales['local']['channels'] * 2 ** (1 - i))
        self.token_upscaler = TokenUpscaler(
            token_size, blocks=scales['inm_regional']['blocks']+2, t_blocks=scales['inm_regional']['t_blocks'], skip_size=skip_size
        )
        self.final_conv = nn.Conv2d(token_size // 2 ** (scales['inm_regional']['blocks'] + 2), 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = super().forward(x)
        tokens = []
        skip = {}
        target_tokens = None
        for s in self.scales:
            _, _, T, H, W = x[s]['x'].shape
            spatial = torch.cat([
                x[s][sp][:, None, None, :, None].expand(-1, 1, T, -1, W) if 'lat' in sp
                else x[s][sp][:, None, None, None, :].expand(-1, 1, T, H, -1)
                for sp in self.scales[s]['spatial']
            ], dim=1)
            temporal = torch.cat([x[s][sp].unsqueeze(1) for sp in self.scales[s]['temporal']], dim=1)
            spatial = torch.cat([x[s]['x'], spatial], dim=1)

            encoded, skip_scale = self.scale_encoders[s](spatial, temporal)
            skip[s] = skip_scale

            B, C_enc, T_enc, H_enc, W_enc = encoded.shape
            token_seq = encoded.permute(0, 2, 3, 4, 1).reshape(B, T_enc * H_enc * W_enc, C_enc)
            token_seq = self.token_encoders[s](token_seq)
            if self.scales[s].get('target', False):
                target_tokens = (len(tokens), len(tokens) + token_seq.shape[1], T_enc, H_enc, W_enc)
            tokens.append(token_seq)

        tokens = torch.cat(tokens, dim=1)
        token_start, token_end, T_enc, H_enc, W_enc = target_tokens
        output = self.transformer(tokens)[:, token_start:token_end, :]
        output = output.reshape(-1, T_enc, H_enc, W_enc, self.token_size).permute(0, 4, 1, 2, 3) # (B, C, T, H, W)

        skip = skip['inm_regional'][::-1] + skip['local'][::-1][-2:]
        for i in range(1, 3):
            B, _, T, _, _ = x['local']['x'].shape
            T2 = x['inm_regional']['x'].shape[2]
            _, C, H, W = skip[-i].shape
            skip[-i] = skip[-i].reshape(B, T, C, H, W).mean(dim=1, keepdim=True).expand(B, T2, C, H, W).reshape(B*T2, C, H, W)

        output = self.token_upscaler(output, skip)
        output = self.final_conv(output).squeeze(1)
        B, _, _, H, W = x['local']['x'].shape
        _, _, T, _, _ = x['inm_regional']['x'].shape
        output = output.reshape(B, T, H, W)
        output = torch.sigmoid(output)

        return output

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 
                           batch_first=True, 
                           num_layers=1)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.projection(last)

@saveable
class SWENetV0(MultiscaleNet):
    def __init__(self, variable_encoders, scales, hidden_dim=64):
        super().__init__(variable_encoders)
        self.scales = scales

        self.encoders = nn.ModuleDict()
        for s in scales:
            self.encoders[s] = TemporalEncoder(scales[s]['channels'], hidden_dim)

        self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = super().forward(x)
        encoded = []
        for s in self.scales:
            _, _, T, H, W = x[s]['x'].shape
            spatial = torch.cat([
                x[s][sp][:, None, None, :, None].expand(-1, 1, T, -1, W) if 'lat' in sp
                else x[s][sp][:, None, None, None, :].expand(-1, 1, T, H, -1)
                for sp in self.scales[s]['spatial']
            ], dim=1)
            temporal = torch.cat([x[s][sp].unsqueeze(1) for sp in self.scales[s]['temporal']], dim=1)
            temporal = temporal[:, :, :, None, None].expand(-1, -1, -1, H, W)
            all_data = torch.cat([x[s]['x'], spatial, temporal], dim=1)
            F = all_data.shape[1]
            all_data = all_data.permute(0, 3, 4, 2, 1).reshape(-1, T, F)
            encoded.append(encoders[s](all_data))

        return None