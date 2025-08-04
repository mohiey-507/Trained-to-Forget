import os
import torch

# --- Environment & Paths ---
SAVE_PATH = "/kaggle/working/models"
RECORDS_DIR = "/kaggle/working/records"
PARAMS_PATH = "/kaggle/working/best_params.json"
TEMP_CHECKPOINT_PATH = "/kaggle/temp/optuna_checkpoints"
TUNE_LOG_PATH = "/kaggle/working/tune.log"
TRAIN_LOG_PATH = "/kaggle/working/train.log"
HAR_DATA_DIR = '/kaggle/input/human-action-recognition-har-dataset/Human Action Recognition'
HAR_TRAIN_DIR = os.path.join(HAR_DATA_DIR, 'train')
HAR_CSV_PATH = os.path.join(HAR_DATA_DIR, 'Training_set.csv')

# --- Global Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
SEED = 42

# --- Model Specific Image Sizing ---
IMAGE_SIZE_REGISTRY = {
    "resnet18": (224, 256), "resnet34": (224, 256),
    "efficientnet_b2": (260, 260), "efficientnet_b3": (300, 300),
}

# --- Tuning Configuration ---
TUNE_EPOCHS = 25
N_TRIALS = 12

# --- Training Configuration ---
EXPERIMENTS = [
    {
        "model_name": "efficientnet_b2", 
        "version": "V3", 
        "epochs": 60,
        "batch_size": 32,
        "learning_rate": 5e-4, # Default fallback LR
        "lr_decay_gamma": 0.8, # Default fallback gamma
    },
]
EARLY_STOPPING_PATIENCE = 5
