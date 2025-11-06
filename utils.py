import torch


def get_best_device():
    """
    Get the best available device for PyTorch computations.

    Returns:
        str: The best available device ('cuda', 'mps', or 'cpu')

    Priority order:
        1. 'cuda' - CUDA GPU if available
        2. 'mps' - Apple Metal Performance Shaders if available
        3. 'cpu' - CPU as final fallback
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'