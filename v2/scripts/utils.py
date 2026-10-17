import torch
import random

def to_device(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: to_device(x, device) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(to_device(x, device) for x in v)
    return v

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def analyze_model(model, max_depth=7):
    total_params = 0
    trainable_params = 0
    
    def print_layer(module, prefix="", depth=0):
        nonlocal total_params, trainable_params
        
        if depth > max_depth:
            return
        
        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += num_params
        trainable_params += num_trainable
        
        name = module.__class__.__name__
        if hasattr(module, '_get_name'):
            name = module._get_name()
        
        parts = [f"{name}"]
        if num_params > 0:
            parts.append(f"params={num_params:,}")
        if num_trainable < num_params:
            parts.append(f"trainable={num_trainable:,}")
        if hasattr(module, 'in_channels'):
            parts.append(f"in={module.in_channels}")
        if hasattr(module, 'out_channels'):
            parts.append(f"out={module.out_channels}")
        
        if hasattr(module, 'kernel_size'):
            parts.append(f"kernel={module.kernel_size}")
        if hasattr(module, 'stride'):
            parts.append(f"stride={module.stride}")
        if hasattr(module, 'padding'):
            parts.append(f"padding={module.padding}")
        
        if depth == 0:
            print(f"{prefix}{' '.join(parts)}")
        else:
            print(f"{prefix}├── {' '.join(parts)}")
        
        children = list(module.named_children())
        for i, (name, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            print_layer(child, child_prefix, depth + 1)
    
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print_layer(model)