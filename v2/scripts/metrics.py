import torch
from saver import saveable

@saveable
class WeightedMSE:
    def __init__(self, target_key, mean_dim=None, mask=[]):
        self.target_key = target_key
        self.mean_dim = mean_dim
        self.mask = mask

    def __call__(self, output, target_dict):
        target = target_dict[self.target_key]
        lat = target_dict['lat']  # (lats,)
        mask = torch.ones_like(target, dtype=torch.bool)
        for key, op, v in self.mask:
            if op == '>':
                mask = mask & (target_dict[key] > v)
            else:
                mask = mask & (target_dict[key] < v)                

        weights = torch.cos(torch.deg2rad(lat))
        weights = weights.view(target.shape[0], 1, target.shape[2], 1)
        weights = weights / weights.mean(dim=2, keepdim=True)
        if self.mean_dim is not None:
            output, target, weights = output.mean(dim=self.mean_dim), target.mean(dim=self.mean_dim), weights.mean(dim=self.mean_dim)

        return (((output - target) ** 2) * weights)[mask].mean()

@saveable
class EnsembleMSE:
    def __init__(self, target_key):
        self.target_key = target_key

    def __call__(self, output, target_dict):
        target = target_dict[self.target_key]
        output, _ = output.sort(dim=1)
        target, _ = target.sort(dim=1)
        return ((output - target) ** 2).mean()

@saveable
class WeightedSoftDiceLoss:
    def __init__(self, target_key, eps=1e-6):
        self.target_key = target_key
        self.eps = eps

    def __call__(self, output, target_dict):
        target = target_dict[self.target_key]
        lat = target_dict['lat']

        weights = torch.cos(torch.deg2rad(lat))
        weights = weights.view(target.shape[0], 1, target.shape[2], 1)
        weights = weights / weights.mean(dim=2, keepdim=True)
        
        intersection = (weights * output * target).sum()
        sum_pred = (weights * output).sum()
        sum_target = (weights * target).sum()
        
        dice = (2 * intersection + self.eps) / (sum_pred + sum_target + self.eps)
        loss = 1 - dice
        
        return loss

class MetricsDict(dict):
    def __init__(self, *args, loss_metric=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_metric = loss_metric

    def __add__(self, other):
        return MetricsDict({k: v + other.get(k, 0) for k, v in self.items()}, loss_metric=self.loss_metric)
    
    def __truediv__(self, other):
        return MetricsDict({k: v / other for k, v in self.items()}, loss_metric=self.loss_metric)
    
    def __iadd__(self, other):
        for k in other: self[k] = self.get(k, 0) + other[k]
        return self
    
    def __itruediv__(self, other):
        for k in self: self[k] /= other
        return self

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return self
        return self.__add__(other)

    def __format__(self, format_spec):
        parts = []
        for k, v in self.items():
            val = v.item() if hasattr(v, 'item') else v
            if format_spec:
                parts.append(f"{k}:{val:{format_spec}}")
            else:
                parts.append(f"{k}:{val}")
        return ", ".join(parts)
    
    def item(self):
        return self

    def backward(self):
        return self[self.loss_metric].backward()

class BaseBinaryMetric:
    def __init__(self, target_key, threshold=0.5):
        self.target_key = target_key
        self.threshold = threshold

    def _get_weights(self, lat, target_shape):
        weights = torch.cos(torch.deg2rad(lat))
        weights = weights.view(target_shape[0], 1, target_shape[2], 1)
        weights = weights / weights.mean(dim=2, keepdim=True)
        return weights

    def _get_counts(self, output, target_dict):
        target = target_dict[self.target_key]
        lat = target_dict['lat']
        weights = self._get_weights(lat, target.shape)
        pred = (output > self.threshold).float()
        TP = (weights * pred * target).sum()
        FP = (weights * pred * (1 - target)).sum()
        FN = (weights * (1 - pred) * target).sum()
        TN = (weights * (1 - pred) * (1 - target)).sum()
        return TP, FP, FN, TN

@saveable
class Precision(BaseBinaryMetric):
    def __call__(self, output, target_dict):
        TP, FP, _, _ = self._get_counts(output, target_dict)
        eps = 1e-6
        return ((TP + eps) / (TP + FP + eps)).item()


@saveable
class Recall(BaseBinaryMetric):
    def __call__(self, output, target_dict):
        TP, _, FN, _ = self._get_counts(output, target_dict)
        eps = 1e-6
        return ((TP + eps) / (TP + FN + eps)).item()


@saveable
class Accuracy(BaseBinaryMetric):
    def __call__(self, output, target_dict):
        TP, FP, FN, TN = self._get_counts(output, target_dict)
        eps = 1e-6
        return ((TP + TN + eps) / (TP + TN + FP + FN + eps)).item()


@saveable
class F1(BaseBinaryMetric):
    def __call__(self, output, target_dict):
        TP, FP, FN, _ = self._get_counts(output, target_dict)
        eps = 1e-6
        precision = (TP + eps) / (TP + FP + eps)
        recall = (TP + eps) / (TP + FN + eps)
        return (2 * precision * recall / (precision + recall + eps)).item()

@saveable
class ComplexMetric:
    def __init__(self, metrics, loss_metric):
        self.metrics = metrics
        self.loss_metric = loss_metric
        self.mode = None

    def set_mode(self, mode):
        self.mode = mode

    def __call__(self, output, target_dict):
        if self.mode == 'train':
            return MetricsDict({self.loss_metric: self.metrics[self.loss_metric](output, target_dict)},
                               loss_metric=self.loss_metric)

        return MetricsDict({name: metric(output, target_dict) for name, metric in self.metrics.items()},
                           loss_metric=self.loss_metric)
