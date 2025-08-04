import os
import sys
import json
import torch
import logging
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import *

def train_model():
    # --- Get Base Experiment Config ---
    exp_config = config.EXPERIMENTS[0]
    model_name = exp_config["model_name"]
    version = exp_config["version"]
    epochs = exp_config["epochs"]
    batch_size = exp_config["batch_size"]
    
    # --- Load Best Hyperparameters ---
    if os.path.exists(config.PARAMS_PATH):
        logging.info(f"Loading best parameters from {config.PARAMS_PATH}")
        with open(config.PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        learning_rate = best_params['learning_rate']
        lr_decay_gamma = best_params['lr_decay_gamma']
    else:
        logging.warning(f"Best parameter file not found at {config.PARAMS_PATH}. Using default values from config.")
        learning_rate = exp_config["learning_rate"]
        lr_decay_gamma = exp_config["lr_decay_gamma"]

    full_model_name = f"{model_name}_{version}_final"
    logging.info(f"--- Starting Final Training Run for {full_model_name} ---")
    logging.info(f"--- Config: LR={learning_rate:.2e}, Gamma={lr_decay_gamma:.2f}, BS={batch_size} ---")

    # --- Setup ---
    set_seed(config.SEED)
    
    # --- Data (with strong augmentations) ---
    crop_size, resize_size = config.IMAGE_SIZE_REGISTRY[model_name]
    transforms = get_strong_augs(crop_size, resize_size)
    
    full_dataset = HumanActivityDataset(config.HAR_CSV_PATH, config.HAR_TRAIN_DIR, transform=transforms)
    num_classes = len(full_dataset.classes)
    
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=config.SEED)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader, val_loader = get_dataloaders(
        train_dataset, val_dataset, batch_size, config.NUM_WORKERS, config.SEED
    )

    # --- Model ---
    model = get_model(model_name, num_classes).to(config.DEVICE)
    unfrozen_layers = get_layers_to_unfreeze(model, model_name, version)
    loss_fn = nn.CrossEntropyLoss()

    # --- Run Final Training ---
    best_val_loss, best_epoch = train(
        model=model, model_name=full_model_name, version=version,
        learning_rate=learning_rate, lr_decay_gamma=lr_decay_gamma,
        train_loader=train_loader, val_loader=val_loader, loss_fn=loss_fn,
        epochs=epochs, device=config.DEVICE, unfrozen_layers=unfrozen_layers,
        save_path=config.SAVE_PATH,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        optuna_trial=None
    )
    
    logging.info(f"--- Training Finished ---")
    logging.info(f"Best validation loss: {best_val_loss:.5f} at epoch {best_epoch}")
    logging.info(f"Final model saved to {os.path.join(config.SAVE_PATH, full_model_name + '_best.pth')}")


if __name__ == '__main__':
    setup_logging()
    if not os.path.exists(config.HAR_CSV_PATH):
        logging.error("Human Action Recognition dataset not found. Skipping training.")
    else:
        train_model()
