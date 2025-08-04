import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from .utils import seed_worker

def get_simple_augs(image_size: int, resize_size: int) -> v2.Compose:
    """Returns a composition of simple augmentations for tuning."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_strong_augs(image_size: int, resize_size: int) -> v2.Compose:
    """Returns a composition of strong augmentations for training."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.5),
        v2.RandomPerspective(distortion_scale=0.1, p=0.3),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    seed: int
) -> tuple[DataLoader, DataLoader]:
    
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
