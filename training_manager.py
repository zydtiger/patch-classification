import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Literal
from pathlib import Path

from early_stopping import EarlyStopping


class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        early_stopping: EarlyStopping,
        logdir: str,
        monitor: Literal["val_loss"] | Literal["val_acc"],
        device: str,
        checkpoint_dir: Path = Path("./checkpoints"),
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(logdir)
        self.model.to(device)
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Early stopping setup
        self.monitor = monitor
        self.early_stopping = early_stopping

        self.start_epoch = 0

    def save_checkpoint(self, epoch: int):
        """Save a checkpoint of the model"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "early_stopping_state": {
                "best_loss": self.early_stopping.best_loss,
                "counter": self.early_stopping.counter,
                "best_model": self.early_stopping.best_model,
            },
        }

        # Save the checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load a checkpoint and restore the training state"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore other training states
        self.start_epoch = checkpoint["epoch"] + 1

        # Restore early stopping state
        self.early_stopping.best_loss = checkpoint["early_stopping_state"]["best_loss"]
        self.early_stopping.counter = checkpoint["early_stopping_state"]["counter"]
        self.early_stopping.best_model = checkpoint["early_stopping_state"][
            "best_model"
        ]

    def train(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
    ):
        for epoch in range(self.start_epoch, num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch, num_classes)
            val_loss, val_acc = self.validate(val_loader, epoch)

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch)

            # Check early stopping
            monitor_value = val_loss if self.monitor == "val_loss" else val_acc
            if self.early_stopping(monitor_value, self.model):
                print(f"\nEarly stopping triggered after epoch {epoch}")
                break

    def train_epoch(
        self, train_loader: DataLoader, epoch: int, num_classes: int
    ) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, targets_ohe, targets) in enumerate(pbar):
            data, targets_ohe, targets = (
                data.to(self.device, non_blocking=True),
                targets_ohe.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
            )

            self.optimizer.zero_grad()
            outputs = self.model(data)
            losses = self.criterion(outputs, targets_ohe)
            losses = losses.mean()

            losses.backward()
            self.optimizer.step()

            running_loss += losses.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            pbar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # Log metrics to tensorboard
        self.writer.add_scalar("Training Loss", epoch_loss, epoch)
        self.writer.add_scalar("Training Accuracy", epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, val_loader: DataLoader, epoch: int) -> tuple[float, float]:
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.inference_mode():
            pbar = tqdm(val_loader, desc=f"Val {epoch}")
            for data, targets_ohe, targets in pbar:
                data, targets_ohe, targets = (
                    data.to(self.device, non_blocking=True),
                    targets_ohe.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )

                outputs = self.model(data)
                losses = self.criterion(outputs, targets_ohe)
                loss = losses.mean()

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                pbar.set_postfix({"acc": 100.0 * correct / total})

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        # Log validation metrics to tensorboard
        self.writer.add_scalar("Validation Loss", val_loss, epoch)
        self.writer.add_scalar("Validation Accuracy", val_acc, epoch)

        return val_loss, val_acc

    def get_best_model(self) -> dict:
        if self.early_stopping.best_model is None:
            raise Exception("Model not trained!")

        return self.early_stopping.best_model
