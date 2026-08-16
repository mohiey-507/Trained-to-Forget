import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

def get_simple_augs(crop_size=224, resize_size=256):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_strong_augs(crop_size=224, resize_size=256):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class MixupCollate:
    """Applies Mixup inside DataLoader and yields (images, soft_labels)."""
    def __init__(self, num_classes, alpha=0.4):
        self.num_classes = num_classes
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        batch_size = images.size(0)
        one_hot_labels = torch.zeros(batch_size, self.num_classes).scatter_(
            1, labels.view(-1, 1), 1
        )
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            index = torch.randperm(batch_size)
            
            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_labels = lam * one_hot_labels + (1 - lam) * one_hot_labels[index, :]
            return mixed_images, mixed_labels
            
        return images, one_hot_labels

def get_dataloaders(
    train_dataset, val_dataset, batch_size, num_workers, seed, 
    use_mixup=False, num_classes=None
):
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_collate = None
    if use_mixup:
        if num_classes is None:
            raise ValueError("num_classes must be provided for Mixup")
        train_collate = MixupCollate(num_classes=num_classes, alpha=0.4)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=g,
        collate_fn=train_collate,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True 
    )
    
    return train_loader, val_loader
