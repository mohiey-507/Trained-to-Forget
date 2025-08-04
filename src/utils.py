import logging
import random
import numpy as np
import torch

def setup_logging():
    """Configures the root logger for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

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
