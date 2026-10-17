import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from saver import saveable, load_from_config_content
from collections import defaultdict
from scales import normalize, denormalize

def get_scale_varname(v):
    if v.startswith('inm_'):
        if '_' not in v[4:] or v.endswith('_clim') and '_' not in v[4:-5]:
            scale = "inm_"
            varname = v[4:]
        else:
            scale = v.split('_')[-1]
            varname = v[4:-len(scale)-1]
            scale = "inm_" + scale
    else:
        if '_' not in v or v.endswith('_clim') and '_' not in v[:-5]:
            scale = ""
            varname = v
        else:
            scale = v.split('_')[-1]
            varname = v[:-len(scale)-1]
    return scale, varname

@saveable
class DummyModel(nn.Module):
    def __init__(self, target_key, size=None, constant=None):
        super().__init__()
        self.target_key = target_key
        self.size = size
        self.constant = constant
        self.zero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        value = x[self.target_key].mean(dim=1) if len(x[self.target_key].shape) == 5 else x[self.target_key]
        if self.size is not None:
            value = F.interpolate(value, self.size, mode='bilinear', align_corners=False)
        if self.constant is not None:
            value = value * 0 + self.constant
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
    def __init__(self, num_stats, ens_size, varname, hidden_dim=8, latent_dim=8):
        super().__init__()
        input_dim = 1
        self.ens_size = ens_size
        self.num_stats = num_stats
        self.hidden_dim = hidden_dim
        self.varname = varname

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
            nn.Linear(num_stats + 2 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.latent_codes = nn.Parameter(torch.randn(ens_size, latent_dim) * 0.1)

    def encode(self, x):
        B, E, T, H, W = x.shape
        N = B * T * H * W
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(N, E, 1) # (N, E, 1)
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

        mean = x_flat.mean(dim=1)
        std = x_flat.std(dim=1)

        return torch.cat([encoded, mean, std], dim=1)

    def decode(self, encoded, shape=None):
#        if shape is None:
        B, E, T, H, W = shape
        N = B * T * H * W

        stats_expanded = encoded.unsqueeze(1).expand(-1, E, -1) # (N, E, num_stats)
        latent = self.latent_codes[:E].unsqueeze(0).expand(N, -1, -1) # (N, E, latent_dim)
        decoder_input = torch.cat([stats_expanded, latent], dim=-1) # (N, E, num_stats + latent_dim)
        output = self.decoder(decoder_input) # (N, E, 1)
        output = output.reshape(B, T, H, W, E, -1).permute(0, 4, 1, 2, 3, 5).squeeze(-1) # (B, E, T, H, W)
        return output

    def forward(self, x):
        x = normalize(list(x.values())[0], self.varname)
        return denormalize(self.decode(self.encode(x), x.shape), self.varname)

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

@saveable
class MultiEnsembleEncoder(nn.Module):
    def __init__(self, variable_encoders):
        super().__init__()
        self._init_variable_encoders(variable_encoders)
        self._freeze_variable_encoders()

    def _get_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_variable_encoders(self, variable_encoders):
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
            checkpoint = torch.load(f'{p}/checkpoints/last.pt', map_location=self._get_device())
            encoder.load_state_dict(checkpoint['model_state_dict'])
            self.variable_encoders[v] = encoder

    def _freeze_variable_encoders(self):
        for encoder in self.variable_encoders.values():
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False

    def encode(self, x):
        result = {}
        for variable in x:
            scale, varname = get_scale_varname(variable)
            xv = normalize(x[variable], varname if "inm_" not in scale else "inm_" + varname)
            if 'inm_' in scale:
                if varname == 'snow_cover':
                    xv = xv.mean(dim=1, keepdim=True)
                if '_clim' in varname:
                    xv = xv.unsqueeze(1)
                if varname in self.variable_encoders and self.variable_encoders[varname] is not None:
                    B, _, T, H, W = xv.shape
                    xv = self.variable_encoders[varname].encode(xv)
                    xv = xv.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)
            else:
                xv = xv.unsqueeze(1)

            result[variable] = xv
        return result

    def decode(self, x, full=True):
        result = {}
        for variable in x:
            scale, varname = get_scale_varname(variable)
            xv = denormalize(x[variable], varname if "inm_" not in scale else "inm_" + varname)
            if varname in self.variable_encoders and self.variable_encoders[varname] is not None:
                B, _, T, H, W = xv.shape
                if full:
                    xv = self.variable_encoders[varname].decode(xv).reshape(B, -1, T, H, W)
            if 'inm_' not in scale:
                xv = xv.squeeze(1)

            result[variable] = xv
        return result

    def forward(self, x):
        return self.encode(x)

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, pool_size=2):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_convs):
            layers.append(nn.Conv1d(ch, out_channels, kernel_size, stride, padding=1, padding_mode='reflect'))
            layers.append(nn.ReLU(inplace=True))
            ch = output_channels
        layers.append(nn.Conv1d(ch, out_channels, kernel_size, pool_size, padding=1, padding_mode='reflect'))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, pool_size=2):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_convs):
            layers.append(nn.Conv2d(ch, out_channels, kernel_size, stride, padding=1, padding_mode='reflect'))
            layers.append(nn.ReLU(inplace=True))
            ch = output_channels
        layers.append(nn.Conv2d(ch, out_channels, kernel_size, pool_size, padding=1, padding_mode='reflect'))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UpscaleBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, relu=True):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0))
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs):
            layers.append(nn.Conv1d(in_channels if i==0 else out_channels, out_channels, kernel_size, stride, padding=1, padding_mode='reflect'))
            layers.append(nn.ReLU(inplace=True))
        if not relu:
            layers = layers[:-1]
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip=None):
        x = self.block(x)
        return x

class UpscaleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, kernel_size=3, stride=1, padding=1, output_padding=0, relu=True):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=padding, output_padding=output_padding))
        layers.append(nn.ReLU(inplace=True))
        for i in range(n_convs):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1, padding_mode='reflect'))
            layers.append(nn.ReLU(inplace=True))
        if not relu:
            layers = layers[:-1]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)      # [B, out_ch, H*2, W*2]
        return x

class FiLMMixin:
    def _init_film_layers(self, modules, cond_dim, out_channels, conv_type):
        self.film_gammas = nn.ModuleList()
        self.film_betas = nn.ModuleList()
        for layer in modules:
            if isinstance(layer, conv_type):
                self.film_gammas.append(nn.Linear(cond_dim, out_channels))
                self.film_betas.append(nn.Linear(cond_dim, out_channels))
        self._init_film_weights()

    def _init_film_weights(self):
        for gamma, beta in zip(self.film_gammas, self.film_betas):
            nn.init.constant_(gamma.weight, 0)
            nn.init.constant_(gamma.bias, 1)
            nn.init.constant_(beta.weight, 0)
            nn.init.constant_(beta.bias, 0)
    
    def _apply_film(self, x, cond, idx):
        gamma = self.film_gammas[idx](cond)
        beta = self.film_betas[idx](cond)
        for _ in range(x.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return x * gamma + beta

class FiLMConvBlock2d(ConvBlock2d, FiLMMixin):
    def __init__(self, in_channels, out_channels, cond_dim, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self._init_film_layers(self.block, cond_dim, out_channels, nn.Conv2d)

    def forward(self, x, cond):
        idx = 0
        for layer in self.block:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self._apply_film(x, cond, idx)
                idx += 1
        return x

class FiLMConvBlock1d(ConvBlock1d, FiLMMixin):
    def __init__(self, in_channels, out_channels, cond_dim, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self._init_film_layers(self.block, cond_dim, out_channels, nn.Conv1d)

    def forward(self, x, cond):
        idx = 0
        for layer in self.block:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                x = self._apply_film(x, cond, idx)
                idx += 1
        return x

class FiLMUpscaleBlock2d(UpscaleBlock2d, FiLMMixin):
    def __init__(self, in_channels, out_channels, cond_dim, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self._init_film_layers(self.block, cond_dim, out_channels, nn.ConvTranspose2d)

    def forward(self, x, cond):
        idx = 0
        for layer in self.block:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                x = self._apply_film(x, cond, idx)
                idx += 1
        return x

class FiLMUpscaleBlock1d(UpscaleBlock1d, FiLMMixin):
    def __init__(self, in_channels, out_channels, cond_dim, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self._init_film_layers(self.block, cond_dim, out_channels, nn.ConvTranspose1d)

    def forward(self, x, cond):
        idx = 0
        for layer in self.block:
            x = layer(x)
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.ConvTranspose1d):
                x = self._apply_film(x, cond, idx)
                idx += 1
        return x

@saveable
class TokenEncoder(nn.Module):
    def __init__(self, in_channels, variables, scale, variable_encoders,
                 token_dim=64, spatial_blocks=3, temporal_blocks=2, kernel_size=3,
                 n_convs_per_block=2, first_conv_dim=32, cond_dim=16, paddings=[]):
        super().__init__()

        self.ensemble_encoder = MultiEnsembleEncoder(variable_encoders)
        self.variables = variables
        self.scale = scale
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.spatial_blocks = spatial_blocks
        self.temporal_blocks = temporal_blocks
        self.cond_dim = cond_dim

        if in_channels > 1:
            self.first_conv = nn.Conv2d(1, in_channels, kernel_size, stride=1, padding=1, padding_mode='reflect')
            self.last_conv = nn.Conv2d(in_channels, 1, kernel_size, stride=1, padding=1, padding_mode='reflect')

        if len(paddings) < spatial_blocks:
            paddings += [(1, 0)]*(spatial_blocks - len(paddings))

        self.spatial_encoder = nn.ModuleList()
        curr_ch = in_channels
        out_ch = first_conv_dim
        for i in range(spatial_blocks):
            self.spatial_encoder.append(
                FiLMConvBlock2d(curr_ch, out_ch, cond_dim, n_convs=n_convs_per_block, kernel_size=kernel_size)
            )
            curr_ch = out_ch
            out_ch = curr_ch * 2

        self.temporal_encoder = nn.ModuleList()
        for i in range(temporal_blocks):
            self.temporal_encoder.append(
                FiLMConvBlock1d(curr_ch, out_ch, cond_dim, n_convs=n_convs_per_block, kernel_size=kernel_size)
            )
            curr_ch = out_ch
            out_ch = curr_ch * 2

        self.fc_mu = nn.Linear(curr_ch, self.token_dim)
        self.fc_logvar = nn.Linear(curr_ch, self.token_dim)
        self.fc_dec = nn.Linear(self.token_dim, curr_ch)

        self.temporal_decoder = nn.ModuleList()
        for i in range(temporal_blocks):
            out_ch = curr_ch // 2
            self.temporal_decoder.append(
                FiLMUpscaleBlock1d(curr_ch, out_ch, cond_dim, n_convs=n_convs_per_block, kernel_size=kernel_size)
            )
            curr_ch = out_ch
        
        self.spatial_decoder = nn.ModuleList()
        for i in range(spatial_blocks - 1):
            out_ch = curr_ch // 2
            self.spatial_decoder.append(
                FiLMUpscaleBlock2d(curr_ch, out_ch, cond_dim, n_convs=n_convs_per_block, kernel_size=kernel_size,
                                   padding=paddings[i][0], output_padding=paddings[i][1])
            )
            curr_ch = out_ch

        self.spatial_decoder.append(FiLMUpscaleBlock2d(curr_ch, self.in_channels, cond_dim, n_convs=n_convs_per_block, relu=False))
        self.var_emb =  nn.Embedding(len(variables), cond_dim)
        self.var_to_idx = {v: i for i, v in enumerate(variables)}

    
    def _get_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode(self, x):
        x = self.ensemble_encoder.encode(x)
        x_variables = []
        for variable in sorted(x.keys()):
            scale, varname = get_scale_varname(variable)
            if scale != self.scale or varname not in self.variables:
                continue

            idx = self.var_to_idx[varname]
            x_variables.append((idx, variable, x[variable].shape[1]))
        
        xs, cond = [], []
        for idx, variable, channels in x_variables:
            cond.append(self.var_emb(torch.tensor(idx, device=self._get_device())).unsqueeze(0))
            if self.in_channels > 1 and channels == 1:
                B, _, T, H, W = x[variable].shape
                x[variable] = x[variable].permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
                x[variable] = self.first_conv(x[variable])
                x[variable] = x[variable].reshape(B, T, self.in_channels, H, W).permute(0, 2, 1, 3, 4)
            xs.append(x[variable].unsqueeze(1))

        x = torch.cat(xs, dim=1)
        cond = torch.cat(cond, dim=0)

        B, V, C, T, H, W = x.shape
        cond = cond.unsqueeze(0).expand(B, V, self.cond_dim).reshape(B * V, self.cond_dim)

        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * V * T, C, H, W)
        cond_spatial = cond.unsqueeze(1).expand(B * V, T, self.cond_dim).reshape(B * V * T, self.cond_dim)
        for block in self.spatial_encoder:
            x = block(x, cond_spatial)

        _, C_enc, H_enc, W_enc = x.shape
        x = x.reshape(B * V, T, C_enc, H_enc, W_enc).permute(0, 3, 4, 2, 1).reshape(B * V * H_enc * W_enc, C_enc, T)
        cond_temporal = cond.unsqueeze(1).unsqueeze(1).expand(B * V, H_enc, W_enc, self.cond_dim).reshape(B * V * H_enc * W_enc, self.cond_dim)
        for block in self.temporal_encoder:
            x = block(x, cond_temporal)

        _, C_enc, T_enc = x.shape
        x = x.reshape(B * V, H_enc, W_enc, C_enc, T_enc).permute(0, 4, 1, 2, 3).reshape(B * V * T_enc * H_enc * W_enc, C_enc)
        mu = self.fc_mu(x)
        if self.training:
            std = torch.exp(0.5 * self.fc_logvar(x))
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        z = z.reshape(B, V * T_enc * H_enc * W_enc, self.token_dim)

        return z, (T_enc, H_enc, W_enc), x_variables
    
    def decode(self, x, shape, variables):
        B = x.shape[0]
        T_enc, H_enc, W_enc = shape
        V = len(variables)

        x = x[:, :, :self.token_dim]
        cond = torch.cat([self.var_emb(torch.tensor(idx, device=self._get_device())).unsqueeze(0) for idx, _, _ in variables], dim=0)
        cond = cond.unsqueeze(0).expand(B, V, self.cond_dim).reshape(B * V, self.cond_dim)

        x = x.reshape(B * V * T_enc * H_enc * W_enc, self.token_dim)
        x = self.fc_dec(x)
        _, C_enc = x.shape

        x = x.reshape(B * V, T_enc, H_enc * W_enc, C_enc).permute(0, 2, 3, 1).reshape(B * V * H_enc * W_enc, C_enc, T_enc)
        cond_temporal = cond.unsqueeze(1).unsqueeze(1).expand(B * V, H_enc, W_enc, self.cond_dim).reshape(B * V * H_enc * W_enc, self.cond_dim)
        for block in self.temporal_decoder:
            x = block(x, cond_temporal)

        _, C_enc, T = x.shape
        x = x.reshape(B * V, H_enc, W_enc, C_enc, T).permute(0, 4, 3, 1, 2).reshape(B * V * T, C_enc, H_enc, W_enc)
        cond_spatial = cond.unsqueeze(1).expand(B * V, T, self.cond_dim).reshape(B * V * T, self.cond_dim)
        for block in self.spatial_decoder:
            x = block(x, cond_spatial)

        _, C, H, W = x.shape
        x = x.reshape(B, V, T, C, H, W).permute(0, 1, 3, 2, 4, 5)

        result = {}
        for i, (_, variable, channels) in enumerate(variables):
            scale, varname = get_scale_varname(variable)
            result[varname] = x[:, i, :, :, :, :]
            if self.in_channels > 1 and channels == 1:
                result[varname] = result[varname].permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                result[varname] = self.last_conv(result[varname])
                result[varname] = result[varname].reshape(B, T, 1, H, W).permute(0, 2, 1, 3, 4)

        return result
    
    def forward(self, x):
        tokens, shape, variables = self.encode(x)
        return self.decode(tokens, shape, variables)


class TokenRefiner(nn.Module):
    def __init__(self, encoders, target_exp, target_var, token_dim=64,
                 n_vars=64, n_scales=16, n_exps=8,
                 max_t=64, max_hw=4096, n_layers=2, n_heads=4):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.target_exp = target_exp
        self.target_var = target_var
        self.exps = list(encoders.keys())
        self.exp_to_idx = {e: i for i, e in enumerate(self.exps)}

        self.coord_emb = nn.Embedding(max_hw, token_dim)
        self.time_emb = nn.Embedding(max_t, token_dim)
        self.var_emb = nn.Embedding(n_vars, token_dim)
        self.scale_emb = nn.Embedding(n_scales, token_dim)
        self.exp_emb = nn.Embedding(n_exps, token_dim)
        self.fuse = nn.Linear(6 * token_dim, token_dim)

        layer = nn.TransformerEncoderLayer(
            token_dim, n_heads, 4 * token_dim,
            batch_first=True, norm_first=True,
            dropout=0.0, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(token_dim, token_dim)

        self._vmap, self._smap = {}, {}

    def _vi(self, v):
        if v not in self._vmap:
            self._vmap[v] = len(self._vmap)
        return self._vmap[v]

    def _si(self, s):
        if s not in self._smap:
            self._smap[s] = len(self._smap)
        return self._smap[s]

    def _prep(self, x, exp, device):
        tok = x[f'tokens_{exp}']
        T, H, W = x[f'shape_{exp}']
        vs = x[f'variables_{exp}']
        V, HW = len(vs), H * W
        N = V * T * HW

        c = torch.arange(HW, device=device).repeat(V * T)
        t = torch.arange(T, device=device).repeat_interleave(HW).repeat(V)

        vi, si = [], []
        tgt = torch.zeros(N, dtype=torch.bool, device=device)
        for i, (_, name, _) in enumerate(vs):
            s, v = get_scale_varname(name)
            vi.append(self._vi(v))
            si.append(self._si(s))
            if exp == self.target_exp and v == self.target_var:
                tgt[i * T * HW:(i + 1) * T * HW] = True

        v = torch.tensor(vi, device=device).repeat_interleave(T * HW)
        s = torch.tensor(si, device=device).repeat_interleave(T * HW)
        e = torch.full((N,), self.exp_to_idx[exp], device=device, dtype=torch.long)
        return tok, c, t, v, s, e, tgt

    def forward(self, x):
        device = next(self.parameters()).device
        toks, cs, ts, vs, ss, es = [], [], [], [], [], []
        target_pos, off = [], 0

        for exp in self.exps:
            tok, c, t, v, s, e, tgt = self._prep(x, exp, device)
            toks.append(tok); cs.append(c); ts.append(t); vs.append(v); ss.append(s); es.append(e)
            if tgt.any():
                target_pos.append(torch.nonzero(tgt).squeeze(-1) + off)
            off += tok.shape[1]

        tokens = torch.cat(toks, dim=1)
        B = tokens.shape[0]
        c, t, v, s, e = (torch.cat(z) for z in (cs, ts, vs, ss, es))

        ctx = torch.cat([
            self.coord_emb(c), self.time_emb(t),
            self.var_emb(v), self.scale_emb(s), self.exp_emb(e),
        ], dim=-1).unsqueeze(0).expand(B, -1, -1)

        h = self.head(self.tr(self.fuse(torch.cat([tokens, ctx], dim=-1))))
        return h[:, torch.cat(target_pos), :]

    def decode(self, x, tokens):
        T, H, W = x[f'shape_{self.target_exp}']
        for idx, name, _ in x[f'variables_{self.target_exp}']:
            if get_scale_varname(name)[1] == self.target_var:
                return self.encoders[self.target_exp].decode(
                    tokens, (T, H, W), [(idx, name, 1)]
                )[self.target_var]
        raise ValueError(f'{self.target_var} not in {self.target_exp}')

@saveable
class TokenDummyBaseline(nn.Module):
    def __init__(self, token_encoders, target_scale, target_var):
        super().__init__()
        self._init_token_encoders(token_encoders)
        self._freeze_token_encoders()
        self.target_scale = target_scale
        self.target_var = target_var
        self.zero = nn.Parameter(torch.zeros(1))

    def _get_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_token_encoders(self, token_encoders):
        self.token_encoders_configs = {}
        self.token_encoders = nn.ModuleDict()
        for v, p in token_encoders.items():
            if not os.path.exists(p):
                self.token_encoders[v] = None
                continue

            with open(f'{p}/model.json', 'r') as f:
                encoder_config = json.load(f)
            self.token_encoders_configs[v] = encoder_config
            encoder = load_from_config_content(encoder_config)
            checkpoint = torch.load(f'{p}/checkpoints/e050.pt', map_location=self._get_device())
            encoder.load_state_dict(checkpoint['model_state_dict'])
            self.token_encoders[v] = encoder

    def _freeze_token_encoders(self):
        for encoder in self.token_encoders.values():
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False

    def decode(self, x):
        token_encoder = self.token_encoders[self.target_scale]
        for i, (idx, name, ch) in enumerate(x[f'variables_{self.target_scale}']):
            if get_scale_varname(name)[1] == self.target_var:
                T, H, W = x[f'shape_{self.target_scale}']
                sel = x[f'tokens_{self.target_scale}'][:, i * T * H * W:(i + 1) * T * H * W, :]
                x = token_encoder.decode(sel, x[f'shape_{self.target_scale}'], [(idx, name, ch)])

        return token_encoder.ensemble_encoder.decode(x, full=False)[self.target_var] + self.zero * 0

    def forward(self, x):
        return self.decode(x)

class SpaceEmbedding(nn.Module):
    def __init__(self, n_scales, dim=64):
        super().__init__()
        in_dim = 5
        self.proj = nn.Linear(in_dim, dim)
        self.scale_emb = nn.Embedding(n_scales, dim)
        self.dim = dim

    def forward(self, x, scale, shape, scale_idx):
        B = None
        cos_lat = cos_lon = sin_lon = None
        T, H_p, W_p = shape

        for key, value in x.items():
            s, varname = get_scale_varname(key)
            if s != scale:
                continue
            if varname == 'cos_lat':
                cos_lat = value
            elif varname == 'cos_lon':
                cos_lon = value
            elif varname == 'sin_lon':
                if 'inm' in scale:
                    sin_lon = value
                else: # Bug
                    sin_lon = value[::2]
            if B is None:
                B = value.shape[0]

        H_size = cos_lat.shape[-1] // H_p
        W_size = cos_lon.shape[-1] // W_p

        cos_lat = cos_lat.view(B, H_p, H_size)
        cos_lon = cos_lon.view(B, W_p, W_size)
        sin_lon = sin_lon.view(B, W_p, W_size)
        lat_span = (cos_lat[..., -1] - cos_lat[..., 0]).abs()
        lon_span = (cos_lon[..., -1] - cos_lon[..., 0]).abs()

        feats = torch.stack([
            cos_lat.mean(dim=-1)[:, :, None].expand(-1, -1, W_p),
            cos_lon.mean(dim=-1)[:, None, :].expand(-1, H_p, -1),
            sin_lon.mean(dim=-1)[:, None, :].expand(-1, H_p, -1),
            lat_span[:, :, None].expand(-1, -1, W_p),
            lon_span[:, None, :].expand(-1, H_p, -1)
        ], dim=-1)

        emb = self.proj(feats)
        sidx = torch.tensor(scale_idx, device=emb.device)
        emb = emb + self.scale_emb(sidx)

        return emb

class TimeEmbedding(nn.Module):
    def __init__(self, n_scales, dim=64):
        super().__init__()
        in_dim = 4
        self.proj = nn.Linear(in_dim, dim)
        self.scale_emb = nn.Embedding(n_scales, dim)
        self.dim = dim

    def forward(self, x, scale, shape, scale_idx):
        T, H_p, W_p = shape
        cos_day = sin_day = year_norm = lead_time = None
        B = None

        for key, value in x.items():
            s, varname = get_scale_varname(key)
            if s != scale:
                continue
            if varname == 'cos_day':
                cos_day = value
            elif varname == 'sin_day':
                sin_day = value
            elif varname == 'year_norm':
                year_norm = value
            elif varname == 'lead_time':
                lead_time = value
            if B is None:
                B = value.shape[0]

        T_size = cos_day.shape[-1] // T
        H_size = W_size = 1

        cos_day = cos_day.view(B, T, T_size).mean(dim=-1)
        sin_day = sin_day.view(B, T, T_size).mean(dim=-1)
        year_norm = year_norm.view(B, T, T_size).mean(dim=-1)
        lead_time = lead_time.view(B, T, T_size).mean(dim=-1)

        feats = torch.stack([
            cos_day[:, :, None, None].expand(-1, -1, H_p, W_p),
            sin_day[:, :, None, None].expand(-1, -1, H_p, W_p),
            year_norm[:, :, None, None].expand(-1, -1, H_p, W_p),
            lead_time[:, :, None, None].expand(-1, -1, H_p, W_p),
        ], dim=-1)

        emb = self.proj(feats)
        sidx = torch.tensor(scale_idx, device=emb.device)
        emb = emb + self.scale_emb(sidx)

        return emb

class VariableEmbedding(nn.Module):
    def __init__(self, n_vars, dim=64):
        super().__init__()
        self.emb = nn.Embedding(n_vars, dim)
        self.dim = dim

    def forward(self, x, scale, shape, var_to_idx, keep=None):
        T, H_p, W_p = shape
        B = x[f'tokens_{scale}'].shape[0]

        vs = x[f'variables_{scale}']
        if keep is not None:
            vs = [vs[j] for j in keep]

        indices = []
        for _, name, _ in vs:
            _, varname = get_scale_varname(name)
            indices.append(var_to_idx[varname])

        idx = torch.tensor(indices, device=x[f'tokens_{scale}'].device)
        V = len(indices)

        emb = self.emb(idx)
        emb = emb.view(1, V, 1, 1, self.dim)
        emb = emb.expand(B, V, T, H_p * W_p, self.dim)
        return emb.reshape(B, V * T * H_p * W_p, self.dim)

@saveable
class TokenRefiner(TokenDummyBaseline):
    def __init__(self, token_encoders, scales, var_to_idx, target_scale, target_var,
                 token_dim=64, n_layers=2, n_heads=4, dropout=0.0):
        super().__init__(token_encoders, target_scale, target_var)
        self.token_dim = token_dim
        d_model = token_dim * 2
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model,
            batch_first=True, norm_first=True,
            dropout=dropout, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

        self.space_emb = SpaceEmbedding(len(token_encoders), token_dim)
        self.time_emb = TimeEmbedding(len(token_encoders), token_dim)
        self.var_emb = VariableEmbedding(max(var_to_idx.values()) + 1, token_dim)
        self.fuse = nn.Linear(3 * token_dim, token_dim)

        self.scale_to_idx = {scale: i for i, scale in enumerate(scales)}
        self.var_to_idx = var_to_idx
        self.scales = scales
        self.token_encoder_keys = list(token_encoders.keys())
    
    def forward(self, x):
        parts = []
        keeps = {}

        for i in range(len(self.scales)):
            scale = self.scales[i]
            encoder_key = self.token_encoder_keys[i]

            vs_all = x[f'variables_{encoder_key}']
            T, H_p, W_p = x[f'shape_{encoder_key}']
            V_all = len(vs_all)
            HW = H_p * W_p

            keep = [
                j for j, (_, name, _) in enumerate(vs_all)
                if get_scale_varname(name)[1] in self.var_to_idx
            ]
            if not keep:
                continue
            keeps[encoder_key] = keep

            tok = x[f'tokens_{encoder_key}']
            B, _, D = tok.shape
            V = len(keep)

            tok = tok.reshape(B, V_all, T * HW, D)
            tok = tok[:, keep].reshape(B, V * T * HW, D)

            sp = self.space_emb(x, scale, x[f'shape_{encoder_key}'], self.scale_to_idx[scale])
            sp = sp.reshape(B, HW, D)
            sp = sp.unsqueeze(1).unsqueeze(2).expand(B, V_all, T, HW, D)
            sp = sp[:, keep].reshape(B, V * T * HW, D)

            tm = self.time_emb(x, scale, x[f'shape_{encoder_key}'], self.scale_to_idx[scale])
            tm = tm.reshape(B, T, HW, D)
            tm = tm.unsqueeze(1).expand(B, V, T, HW, D).reshape(B, V * T * HW, D)

            vr = self.var_emb(x, encoder_key, x[f'shape_{encoder_key}'],
                              self.var_to_idx, keep=keep)

            ctx = self.fuse(torch.cat([sp, tm, vr], dim=-1))
            parts.append(torch.cat([tok, ctx], dim=-1))

        merged = self.tr(torch.cat(parts, dim=1))

        off = 0
        for s in self.token_encoder_keys:
            if s not in keeps:
                continue
            keep = keeps[s]
            vs_all = x[f'variables_{s}']
            T, H_p, W_p = x[f'shape_{s}']
            V_all, V = len(vs_all), len(keep)
            HW = H_p * W_p
            n = V * T * HW

            new_tok = merged[:, off:off + n]
            off += n

            full = x[f'tokens_{s}']
            Bf, Nf, Df = full.shape
            full = full.reshape(Bf, V_all, T * HW, Df).clone()
            full[:, keep] = new_tok[:, :, :64].reshape(Bf, V, T * HW, Df)
            x[f'tokens_{s}'] = full.reshape(Bf, V_all * T * HW, Df)

        return self.decode(x)