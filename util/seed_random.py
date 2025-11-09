import torch
import numpy as np
import random
import os

def seed_random(seed = 42):
    # 1. Set all random seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. Force deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. Optional: Ensure reproducibility in data loaders
    # (if using DataLoader with multiple workers)
    torch.use_deterministic_algorithms(True)
