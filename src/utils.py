import logging
import random
import numpy as np
import torch
from typing import Optional

def setup_logging(log_file: Optional[str] = None):
    """Configures the root logger for the project."""
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log file each run
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
