import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
)
from typing import List

def get_model(model_name: str, num_classes: int, dropout_p: float = 0.3) -> nn.Module:
    """
    Loads a pre-trained model and adapts it for fine-tuning.
    """
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
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_features // 4),
            nn.Dropout(dropout_p),
            nn.Linear(in_features // 4, num_classes)
        )
    elif 'eff' in model_name.lower():
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_features // 4),
            nn.Dropout(dropout_p),
            nn.Linear(in_features // 4, num_classes)
        )
    return model

def get_layers_to_unfreeze(model: nn.Module, model_name: str, version: str) -> List[nn.Module]:
    """
    Returns a list of layer groups to unfreeze based on the version.
    """
    layers = []
    if 'res' in model_name.lower():
        base_layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        version_map = {"V1": [], "V2": base_layers[:1], "V3": base_layers[:2], "V4": base_layers[:3]}
        layers = version_map.get(version, [])
    elif 'eff' in model_name.lower():
        base_layers = list(model.features)[::-1]
        version_map = {"V1": [], "V2": base_layers[:1], "V3": base_layers[:2], "V4": base_layers[:4]}
        layers = version_map.get(version, [])

    for layer_group in layers:
        for param in layer_group.parameters():
            param.requires_grad = True
    return layers

def get_optimizer(
    model: nn.Module, model_name: str, unfrozen_layers: List[nn.Module],
    base_lr: float, lr_decay_gamma: float, weight_decay: float
) -> optim.Optimizer:
    """
    Creates an AdamW optimizer with discriminative learning rates for the unfrozen layers.
    """
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
    """
    Sets the model to eval mode, then selectively sets the classifier and
    unfrozen backbone layers back to train mode.
    """
    model.eval()
    if 'res' in model_name.lower():
        model.fc.train()
    elif 'eff' in model_name.lower():
        model.classifier.train()
    for layer_group in unfrozen_layers:
        layer_group.train()
