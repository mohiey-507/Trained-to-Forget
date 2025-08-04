import os
import torch
import optuna
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Tuple

from .model import set_selective_eval_mode, get_optimizer

def train_step(
    model: nn.Module, model_name: str, unfrozen_layers: List[nn.Module],
    dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Performs a single training epoch."""
    set_selective_eval_mode(model, model_name, unfrozen_layers)
    train_loss, train_acc = 0.0, 0.0
    pbar = tqdm(dataloader, desc="[Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
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
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            y_true = labels if labels.ndim == 1 else labels.argmax(dim=1)
            batch_acc = (y_pred_class == y_true).sum().item() / len(outputs)
            val_acc += batch_acc
            pbar.set_postfix({"Loss": f"{loss.item():.5f}", "Acc": f"{batch_acc:.5f}"})
    return val_loss / len(dataloader), val_acc / len(dataloader)

def train(
    model: nn.Module, model_name: str, version: str,
    learning_rate: float, lr_decay_gamma: float,
    train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
    loss_fn: nn.Module, epochs: int, device: torch.device, 
    unfrozen_layers: List[nn.Module], save_path: str, 
    early_stopping_patience: Optional[int] = None,
    optuna_trial: Optional[optuna.Trial] = None
) -> Tuple[float, int]:
    base_model_name = model_name.split('_')[0]
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    optimizer = get_optimizer(
        model, base_model_name, unfrozen_layers,
        base_lr=learning_rate, lr_decay_gamma=lr_decay_gamma
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.4)
    
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_step(
            model, base_model_name, unfrozen_layers, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = validate_step(model, val_loader, loss_fn, device)
        
        scheduler.step(val_loss)
        
        logging.info(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f}")
        
        # --- Pruning and Early Stopping ---
        # Optuna Pruning
        if optuna_trial:
            optuna_trial.report(val_loss, epoch)
            if optuna_trial.should_prune():
                logging.info("Trial pruned by Optuna.")
                return float('inf'), -1
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            full_save_path = os.path.join(save_path, f"{model_name}_best.pth")
            torch.save(model.state_dict(), full_save_path)
            logging.info(f"Best model saved to {full_save_path} (Val Loss: {best_val_loss:.5f})")
        else:
            epochs_no_improve += 1

        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
            break

    logging.info("Training finished.")
    return best_val_loss, best_epoch