import os
import sys
import json
import torch
import optuna
import logging
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import *

def objective(trial: optuna.Trial) -> float:
    """The objective function for Optuna."""
    # --- Suggest Hyperparameters ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    lr_decay_gamma = trial.suggest_float("lr_decay_gamma", 0.7, 0.95)

    # --- Get Base Experiment Config ---
    exp_config = config.EXPERIMENTS[0]
    model_name = exp_config["model_name"]
    version = exp_config["version"]
    batch_size = exp_config["batch_size"]
    
    trial_name = f"{model_name}_{version}_trial_{trial.number}"
    logging.info(f"--- Starting Trial {trial.number}/{config.N_TRIALS} for {trial_name} ---")
    logging.info(f"--- Params: LR={learning_rate:.2e}, Gamma={lr_decay_gamma:.2f} ---")

    # --- Setup ---
    set_seed(config.SEED + trial.number)
    
    # --- Data ---
    crop_size, resize_size = config.IMAGE_SIZE_REGISTRY[model_name]
    transforms = get_simple_augs(crop_size, resize_size)
    
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

    # --- Run Training for Tuning ---
    best_val_loss, _ = train(
        model=model, model_name=trial_name, version=version,
        learning_rate=learning_rate, lr_decay_gamma=lr_decay_gamma,
        train_loader=train_loader, val_loader=val_loader, loss_fn=loss_fn,
        epochs=config.TUNE_EPOCHS, device=config.DEVICE, unfrozen_layers=unfrozen_layers,
        save_path=config.TEMP_CHECKPOINT_PATH,
        optuna_trial=trial,
        early_stopping_patience=None
    )
    
    return best_val_loss

if __name__ == '__main__':
    setup_logging(log_file=config.TUNE_LOG_PATH)
    if not os.path.exists(config.HAR_CSV_PATH):
        logging.error("Human Action Recognition dataset not found. Skipping tuning.")
    else:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config.N_TRIALS)

        logging.info("--- Optuna Hyperparameter Search Finished ---")
        logging.info(f"Best trial validation loss: {study.best_value:.5f}")
        logging.info("Best parameters found: ")
        for key, value in study.best_trial.params.items():
            logging.info(f"  - {key}: {value}")
        
        # --- Save the best parameters to disk ---
        logging.info(f"Saving best parameters to {config.PARAMS_PATH}")
        with open(config.PARAMS_PATH, 'w') as f:
            json.dump(study.best_trial.params, f, indent=4)
        logging.info("Best parameters saved successfully.")
