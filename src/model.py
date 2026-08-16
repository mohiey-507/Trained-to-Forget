import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
)
from typing import List, Dict

def replace_inplace_relu(model: nn.Module):
    """Recursively replaces all inplace ReLU layers with out-of-place ReLUs."""
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_inplace_relu(child)

def get_model(model_name: str, num_classes: int, dropout_p: float = 0.3) -> nn.Module:
    """Loads a pre-trained model and adapts it for fine-tuning."""
    model_registry = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
    }
    model_constructor, weights = model_registry[model_name.lower()]
    model = model_constructor(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    if 'res' in model_name.lower():
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, in_features // 2),  
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(in_features // 2),        
            nn.Dropout(dropout_p),
            nn.Linear(in_features // 2, num_classes) 
        )
    elif 'eff' in model_name.lower():
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout_p),
            nn.Linear(in_features // 2, num_classes)
        )
    
    replace_inplace_relu(model)
    return model

def get_unfreezing_schedule(model: nn.Module, model_name: str, version: str) -> Dict[int, List[nn.Module]]:
    """Returns a dictionary defining the progressive unfreezing schedule."""
    schedules = {}
    if 'resnet' in model_name.lower():
        base_layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        schedules = {
            "V1": {},  
            "V2": {3: [base_layers[0]]},
            "V3": {3: [base_layers[0]], 8: [base_layers[1]], 12: [base_layers[2]]},
        }
    elif 'eff' in model_name.lower():
        base_layers = list(model.features) 
        schedules = {
            "V1": {},
            "V2": {3: [base_layers[8], base_layers[7]]},
            "V3": {3: [base_layers[8], base_layers[7]], 8: [base_layers[6]], 12: [base_layers[5]]},
        }
    return schedules.get(version, {})

def get_optimizer(
    model: nn.Module, model_name: str, unfrozen_layers: List[nn.Module],
    base_lr: float, lr_decay_gamma: float = 0.8, weight_decay: float = 1e-4
) -> optim.Optimizer:
    """Creates an AdamW optimizer with discriminative learning rates."""
    param_groups = []
    
    if 'res' in model_name.lower():
        param_groups.append({'params': model.fc.parameters(), 'lr': base_lr})
    elif 'eff' in model_name.lower():
        param_groups.append({'params': model.classifier.parameters(), 'lr': base_lr})

    for i, layer_group in enumerate(unfrozen_layers):
        lr = base_lr * (lr_decay_gamma ** (i + 1))
        param_groups.append({'params': layer_group.parameters(), 'lr': lr})
        
    return optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay) 

def set_selective_eval_mode(model: nn.Module, model_name: str, unfrozen_layers: List[nn.Module]):
    model.eval()
    if 'res' in model_name.lower():
        model.fc.train()
    elif 'eff' in model_name.lower():
        model.classifier.train()
    
    for layer_group in unfrozen_layers:
        layer_group.train()
