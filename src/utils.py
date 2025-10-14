import logging
import random
import numpy as np
import torch
from typing import Optional
import os
from . import config

def setup_logging(log_file: Optional[str] = None):
    """Configures the root logger for the project."""
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")

def set_seed(seed: int = 42, seed_torch: bool = True):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    logging.info(f'Random seed {seed} has been set.')

def seed_worker(worker_id: int):
    """Seeds worker processes for dataloading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_experiment_config() -> dict:
    """Loads the experiment configuration based on an environment variable."""
    exp_name = os.environ.get('KAGGLE_EXPERIMENT_NAME', config.ACTIVE_EXPERIMENT_NAME)
    logging.info(f"Loading configuration for experiment: '{exp_name}'")
    try:
        exp_config = config.EXPERIMENTS[exp_name]
        return exp_config
    except KeyError:
        logging.error(f"FATAL: Experiment '{exp_name}' not found in config.py.")
        raise