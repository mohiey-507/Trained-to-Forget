import os
import random
import numpy as np
import torch
import logging

def setup_logging(log_file=None):
    """Sets up logging to console and optionally a file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Global seed set to: {seed} (CuDNN Benchmark Enabled)")

def load_experiment_config():
    """Helper to load config based on environment variable."""
    from . import config
    exp_name = os.environ.get('KAGGLE_EXPERIMENT_NAME', config.ACTIVE_EXPERIMENT_NAME)
    if exp_name not in config.EXPERIMENTS:
        raise ValueError(f"Experiment '{exp_name}' not found in config.")
    return config.EXPERIMENTS[exp_name]