import torch
from saver import saveable

@saveable
class WeightedMSE:
    def __init__(self, target_key):
        self.target_key = target_key

    def __call__(self, output, target_dict):
        target = target_dict[self.target_key]
        lat = target_dict['lat']  # (lats,)

        weights = torch.cos(torch.deg2rad(lat))
        weights = weights.view(target.shape[0], 1, target.shape[2], 1)

        return (((output - target) ** 2) * weights).mean()

@saveable
class EnsembleMSE:
    def __init__(self, target_key):
        self.target_key = target_key

    def __call__(self, output, target_dict):
        target = target_dict[self.target_key]
        output, _ = output.sort(dim=1)
        target, _ = target.sort(dim=1)
        return ((output - target) ** 2).mean()