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
    exp_config = load_experiment_config()
    
    model_name = exp_config["model_name"]
    version = exp_config["version"]
    epochs = exp_config["epochs"]
    batch_size = exp_config["batch_size"]
    use_mixup = exp_config["use_mixup"]
    save_every_epoch = exp_config["save_every_epoch"]
    
    if os.path.exists(config.PARAMS_PATH):
        logging.info(f"Loading best parameters from {config.PARAMS_PATH}")
        with open(config.PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        learning_rate = best_params['learning_rate']
        lr_decay_gamma = best_params['lr_decay_gamma']
    else:
        logging.warning(f"Best parameter file not found. Using default values.")
        learning_rate = exp_config["learning_rate"]
        lr_decay_gamma = exp_config["lr_decay_gamma"]

    full_model_name = f"{os.environ.get('KAGGLE_EXPERIMENT_NAME', config.ACTIVE_EXPERIMENT_NAME)}_final"
    logging.info(f"--- Starting Final Training Run for {full_model_name} ---")

    set_seed(config.SEED)
    
    crop_size, resize_size = config.IMAGE_SIZE_REGISTRY[model_name]
    transforms = get_strong_augs(crop_size, resize_size)
    
    full_dataset = HumanActivityDataset(config.HAR_CSV_PATH, config.HAR_TRAIN_DIR, transform=transforms)
    num_classes = len(full_dataset.classes)
    
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=config.SEED)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader, val_loader = get_dataloaders(
        train_dataset, val_dataset, batch_size, config.NUM_WORKERS, config.SEED,
        use_mixup=use_mixup, num_classes=num_classes
    )

    model = get_model(model_name, num_classes).to(config.DEVICE)
    unfrozen_layers = get_layers_to_unfreeze(model, model_name, version)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss, best_epoch = train(
        model=model, model_name=full_model_name, version=version,
        learning_rate=learning_rate, lr_decay_gamma=lr_decay_gamma,
        train_loader=train_loader, val_loader=val_loader, loss_fn=loss_fn,
        epochs=epochs, device=config.DEVICE, unfrozen_layers=unfrozen_layers,
        save_path=config.SAVE_PATH,
        records_dir=config.RECORDS_DIR,
        save_every_epoch=save_every_epoch,
        early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE,
        optuna_trial=None
    )
    
    logging.info(f"--- Training Finished ---")
    logging.info(f"Best validation loss: {best_val_loss:.5f} at epoch {best_epoch}")

if __name__ == '__main__':
    setup_logging(log_file=config.TRAIN_LOG_PATH)
    if not os.path.exists(config.HAR_CSV_PATH):
        logging.error("Dataset not found. Skipping training.")
    else:
        train_model()
