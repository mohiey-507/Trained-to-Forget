import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
from tqdm import tqdm
from typing import Optional, List, Tuple
from torch.amp import GradScaler, autocast

from .model import set_selective_eval_mode, get_optimizer, get_unfreezing_schedule

def train_step(
    model: nn.Module, model_name: str, unfrozen_layers: List[nn.Module],
    dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer,
    device: torch.device, scaler: GradScaler
) -> Tuple[float, float]:
    """Performs a single training epoch."""
    set_selective_eval_mode(model, model_name, unfrozen_layers)
    train_loss, train_acc = 0.0, 0.0
    pbar = tqdm(dataloader, desc="[Train]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        y_true = labels if labels.ndim == 1 else labels.argmax(dim=1)
        batch_acc = (y_pred_class == y_true).sum().item() / len(outputs)
        train_acc += batch_acc
        
        pbar.set_postfix({"Loss": f"{loss.item():.5f}", "Acc": f"{batch_acc:.5f}"})
        
    return train_loss / len(dataloader), train_acc / len(dataloader)

def validate_step(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Performs a single validation epoch."""
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    pbar = tqdm(dataloader, desc="[Val]")
    
    with torch.inference_mode():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
            val_loss += loss.item()
            
            y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            y_true = labels if labels.ndim == 1 else labels.argmax(dim=1)
            batch_acc = (y_pred_class == y_true).sum().item() / len(outputs)
            val_acc += batch_acc
            pbar.set_postfix({"Loss": f"{loss.item():.5f}", "Acc": f"{batch_acc:.5f}"})
            
    return val_loss / len(dataloader), val_acc / len(dataloader)

def get_scheduler(optimizer: optim.Optimizer, total_epochs: int, warmup_epochs: int = 1):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    return torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, main_scheduler])

def train(
    model: nn.Module, model_name: str, version: str,
    learning_rate: float, lr_decay_gamma: float, weight_decay: float,
    train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
    loss_fn: nn.Module, epochs: int, device: torch.device,
    unfrozen_layers: List[nn.Module], save_path: str, 
    records_dir: Optional[str] = None,
    save_every_epoch: bool = False,
    early_stopping_patience: Optional[int] = None,
    optuna_trial: Optional[optuna.Trial] = None,
    lr_stage_decay: float = 0.75, warmup_epochs: int = 1
) -> Tuple[float, int]:
    base_model_name = model_name.split('_')[0]
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    records = []

    schedule = get_unfreezing_schedule(model, base_model_name, version)
    current_lr = learning_rate
    scaler = GradScaler()
    
    optimizer = get_optimizer(
        model, base_model_name, unfrozen_layers,
        base_lr=current_lr, lr_decay_gamma=lr_decay_gamma, weight_decay=weight_decay
    )
    scheduler = get_scheduler(optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs)
    
    os.makedirs(save_path, exist_ok=True)
    if records_dir:
        os.makedirs(records_dir, exist_ok=True)

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")

        if epoch + 1 in schedule:
            new_layers_to_unfreeze = schedule[epoch + 1]
            logging.info(f"Epoch {epoch+1}: Unfreezing {len(new_layers_to_unfreeze)} layer group(s)...")
            
            for layer_group in new_layers_to_unfreeze:
                for param in layer_group.parameters():
                    param.requires_grad = True
            
            unfrozen_layers.extend(new_layers_to_unfreeze)
            current_lr *= lr_stage_decay
            logging.info(f"Re-init Optimizer. New Base LR: {current_lr:.2e}")
            
            optimizer = get_optimizer(
                model, base_model_name, unfrozen_layers, 
                base_lr=current_lr, lr_decay_gamma=lr_decay_gamma, weight_decay=weight_decay
            )
            scheduler = get_scheduler(
                optimizer, total_epochs=epochs - epoch, warmup_epochs=warmup_epochs
            )

        train_loss, train_acc = train_step(
            model, base_model_name, unfrozen_layers, train_loader, loss_fn, optimizer, device, scaler
        )
        val_loss, val_acc = validate_step(model, val_loader, loss_fn, device)
        scheduler.step()
        
        logging.info(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f}")
        
        records.append({
            'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        if optuna_trial:
            optuna_trial.report(val_loss, epoch)
            if optuna_trial.should_prune():
                logging.info("Trial pruned by Optuna.")
                return float('inf'), -1
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(save_model.state_dict(), os.path.join(save_path, f"{model_name}_best.pth"))
            logging.info(f"Best model saved (Val Loss: {best_val_loss:.5f})")
        elif early_stopping_patience:
            epochs_no_improve += 1

        if save_every_epoch:
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(save_model.state_dict(), os.path.join(save_path, f"{model_name}_epoch_{epoch+1}.pth"))
            logging.info(f"Epoch {epoch+1} checkpoint saved.")

        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    logging.info("Training finished.")
    if records_dir:
        pd.DataFrame(records).to_csv(os.path.join(records_dir, f"{model_name}_records.csv"), index=False)
    return best_val_loss, best_epoch