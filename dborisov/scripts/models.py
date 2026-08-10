import torch
import torch.nn as nn 

class SWEUNet(nn.Module):
    """
    SKETCH of the first-experiment model.
    Everything below is selected by the experiment JSON.
    """

    def __init__(self, cfg):
        super().__init__()
        # one front-end per declared stream
        self.fronts = nn.ModuleDict()
        for s in cfg.streams:                       # s.name, s.source, s.channels, s.t, s.scale
            self.fronts[s.name] = StreamFrontEnd(
                has_ensemble = (s.source == "inmcm"),
                has_time     = (s.t is not None),
                e_pool       = build_e_pool(cfg.e_pool),        # mean_std | quantile | deepsets | pma
                t_collapse   = build_t_collapse(cfg.t_collapse),# flatten   | temporal_attn | conv3d
                in_ch=s.channels, t=s.t, stem_ch=cfg.stem_ch,
                work_grid=cfg.work_grid,            # None for full-domain; (H0,W0) for multi-scale
            )
        self.unet = UNet(
            in_ch = cfg.stem_ch * len(cfg.streams),
            base  = cfg.base, depth = cfg.depth,
            bottleneck_attention = cfg.attention,   # heads, dim (e.g. dim=32)
        )
        self.head = nn.Conv2d(cfg.base, 1, kernel_size=1)

    def forward(self, batch):                       # batch: dict[str, Tensor]
        feats = [self.fronts[name](batch[name]) for name in self.fronts]  # each -> [B, F, H, W]
        x = torch.cat(feats, dim=1)                 # [B, F_total, H, W]
        x = self.unet(x)                            # [B, base,   H, W]
        return self.head(x).squeeze(1)              # [B, H, W]  SWE anomaly


class StreamFrontEnd(nn.Module):
    def forward(self, x):
        if self.has_ensemble:                       # [B, E, C, T, H, W]
            x = self.e_pool(x)                      # -> [B, C*k, T, H, W]   (E gone)
        if self.has_time:                           # [B, C', T, H, W]
            x = self.t_collapse(x)                  # -> [B, C'',     H, W]
        x = self.stem(x)                            # -> [B, F,       H, W]
        if self.work_grid is not None:
            x = resize(x, self.work_grid)           # multi-scale only
        return x