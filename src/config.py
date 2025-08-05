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
    "resnet18": (224, 256),
    "efficientnet_b2": (260, 260),
}

# --- Tuning & Training Configuration ---
TUNE_EPOCHS = 25
N_TRIALS = 40
DEFAULT_EARLY_STOPPING_PATIENCE = 5

# --- Experiment Matrix ---
ACTIVE_EXPERIMENT_NAME = "effnet_v3"

EXPERIMENTS = {
    # --- EfficientNet Experiments ---
    "effnet_v1": {
        "model_name": "efficientnet_b2", "version": "V1", "epochs": 50,
        "batch_size": 32, "learning_rate": 1e-03, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": False,
        "weight_decay": 1.5e-5, "dropout_p": 0.40
    },
    "effnet_v2": {
        "model_name": "efficientnet_b2", "version": "V2", "epochs": 50,
        "batch_size": 32, "learning_rate": 1e-03, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": False,
        "weight_decay": 1.5e-5, "dropout_p": 0.40
    },
    "effnet_v3": {
        "model_name": "efficientnet_b2", "version": "V3", "epochs": 50,
        "batch_size": 32, "learning_rate": 1e-03, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": False,
        "weight_decay": 1.5e-5, "dropout_p": 0.40
    },
    "effnet_v3_checkpointed": {
        "model_name": "efficientnet_b2", "version": "V3", "epochs": 50,
        "batch_size": 32, "learning_rate": 1e-03, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": True,
        "weight_decay": 1.5e-5, "dropout_p": 0.40
    },
    
    # --- ResNet Experiments ---
    "resnet_v1": {
        "model_name": "resnet18", "version": "V1", "epochs": 50,
        "batch_size": 32, "learning_rate": 1.6e-04, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": False,
        "weight_decay": 3.75e-3, "dropout_p": 0.40
    },
    "resnet_v2": {
        "model_name": "resnet18", "version": "V2", "epochs": 50,
        "batch_size": 32, "learning_rate": 1.6e-04, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": False,
        "weight_decay": 3.75e-3, "dropout_p": 0.40
    },
    "resnet_v3": {
        "model_name": "resnet18", "version": "V3", "epochs": 50,
        "batch_size": 32, "learning_rate": 1.6e-04, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": False,
        "weight_decay": 3.75e-3, "dropout_p": 0.40
    },
    "resnet_v3_checkpointed": {
        "model_name": "resnet18", "version": "V3", "epochs": 50,
        "batch_size": 32, "learning_rate": 1.6e-04, "lr_decay_gamma": 0.70,
        "use_mixup": True, "save_every_epoch": True,
        "weight_decay": 3.75e-3, "dropout_p": 0.40
    },
}