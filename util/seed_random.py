import torch
import numpy as np
import random
import os

# NOTE: For absolute bit-for-bit reproducibility prefer CPU training.
# Some GPU kernels remain nondeterministic across PyTorch/CUDA versions.
def seed_random(seed: int = 42, device: str = "cpu"):
    """
    Set seeds and deterministic flags. Call once before model creation and again with
    controlled offsets inside loops to ensure repeatable noise & sampling.
    device: 'cpu' or 'cuda'
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms where possible
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # disable TF32 for reproducibility
    if hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    # Recommend warn_only=True during development to avoid exceptions for ops with no deterministic implementation.
    # When you are confident the code supports deterministic ops you can set warn_only=False to raise on nondet ops.
    torch.use_deterministic_algorithms(True, warn_only=True)
