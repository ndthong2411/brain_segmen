import os, random, numpy as np, torch

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Enable cuDNN auto-tuning for A100 optimization
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for A100 Tensor Cores (faster matmul)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
