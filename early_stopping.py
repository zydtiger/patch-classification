import torch.nn as nn


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int, min_delta: float):
        """
        patience (int): How many epochs to wait before stopping when loss is not improving
        min_delta (float): Minimum change in monitored value to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None

    def __call__(self, current_value: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = current_value
            self.best_model = self._get_model_copy(model)
            return False

        if current_value < self.best_loss - self.min_delta:
            self.best_loss = current_value
            self.counter = 0
            self.best_model = self._get_model_copy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False

    def _get_model_copy(self, model: nn.Module) -> dict:
        """Returns a deep copy of the model state"""
        return {
            key: val.cpu().clone().detach() for key, val in model.state_dict().items()
        }
