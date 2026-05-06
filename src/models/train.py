"""
train.py - Training loop with MLflow experiment tracking.

Implements a full training pipeline with:
- Configurable optimizer and learning rate scheduler
- Early stopping with patience
- MLflow logging of parameters, metrics, and model artifacts
- Model checkpointing (best and last)
- Gradient clipping for training stability
"""

import time
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping mechanism to halt training when validation loss stops improving.

    Monitors a specified metric and stops training after a configurable number
    of epochs with no improvement. Also restores the best model weights.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        monitor: Metric name to monitor.
        mode: 'min' for loss (lower is better), 'max' for accuracy.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_weights: Optional[dict] = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current value of the monitored metric.
            model: The model whose weights to save if this is the best score.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        improved = (
            (score < self.best_score - self.min_delta) if self.mode == "min"
            else (score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = score
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience} "
                         f"(best {self.monitor}={self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best(self, model: nn.Module) -> None:
        """Restore model weights to the best observed during training."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class Trainer:
    """
    Training pipeline for healthcare prediction models.

    Handles the full training lifecycle including optimizer setup, learning
    rate scheduling, early stopping, MLflow tracking, and model checkpointing.
    Supports both tabular (MLP) and sequence (LSTM/Transformer) models.

    Attributes:
        model: PyTorch model to train.
        config: Training configuration dictionary.
        device: Torch device for computation.
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: Optional[dict] = None,
        global_config: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model: PyTorch model instance.
            model_config: Model configuration from model_config.yaml.
            global_config: Global configuration from config.yaml.
            device: Override device. If None, auto-detects.
        """
        if model_config is None:
            model_config = load_config("configs/model_config.yaml")
        if global_config is None:
            global_config = load_config("configs/config.yaml")

        self.model = model
        self.model_config = model_config
        self.global_config = global_config
        self.train_config = model_config.get("training", {})

        # Device setup
        if device is not None:
            self.device = device
        else:
            device_str = global_config.get("device", "auto")
            if device_str == "auto":
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu"
                )
            else:
                self.device = torch.device(device_str)

        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Setup components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.early_stopping = self._create_early_stopping()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer from configuration."""
        opt_cfg = self.model_config.get("optimizer", {})
        opt_type = opt_cfg.get("type", "Adam")
        lr = self.train_config.get("learning_rate", 0.001)
        weight_decay = self.train_config.get("weight_decay", 1e-5)

        if opt_type == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            )
        elif opt_type == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_type == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=opt_cfg.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create the learning rate scheduler from configuration."""
        sched_cfg = self.train_config.get("scheduler", {})
        sched_type = sched_cfg.get("type", "ReduceLROnPlateau")

        if sched_type == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_cfg.get("mode", "min"),
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 5),
                min_lr=sched_cfg.get("min_lr", 1e-6),
            )
        elif sched_type == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.get("epochs", 50),
                eta_min=sched_cfg.get("min_lr", 1e-6),
            )
        elif sched_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("factor", 0.5),
            )

        return None

    def _create_criterion(self) -> nn.Module:
        """Create the loss function from configuration."""
        loss_cfg = self.model_config.get("loss", {})
        loss_type = loss_cfg.get("type", "BCEWithLogitsLoss")

        if loss_type == "BCEWithLogitsLoss":
            pos_weight = loss_cfg.get("pos_weight")
            kwargs = {}
            if pos_weight is not None:
                kwargs["pos_weight"] = torch.tensor(pos_weight, device=self.device)
            return nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(label_smoothing=loss_cfg.get("label_smoothing", 0.0))
        else:
            raise ValueError(f"Unknown loss: {loss_type}")

    def _create_early_stopping(self) -> EarlyStopping:
        """Create the early stopping callback from configuration."""
        es_cfg = self.train_config.get("early_stopping", {})
        return EarlyStopping(
            patience=es_cfg.get("patience", 10),
            min_delta=es_cfg.get("min_delta", 0.0001),
            monitor=es_cfg.get("monitor", "val_loss"),
            mode=es_cfg.get("mode", "min"),
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, list[float]]:
        """
        Execute the full training loop.

        Runs epoch-level training with validation, early stopping, learning
        rate scheduling, and MLflow tracking. Returns the training history
        for visualization and analysis.

        Args:
            X_train: Training features. Shape (n, ...) depends on model type.
            y_train: Training labels. Shape (n,) for binary classification.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            lists tracking metrics across epochs.
        """
        batch_size = self.train_config.get("batch_size", 64)
        epochs = self.train_config.get("epochs", 50)

        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)

        # History tracking
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        # MLflow tracking
        mlflow_cfg = self.global_config.get("mlflow", {})
        use_mlflow = mlflow_cfg.get("tracking_uri") is not None

        if use_mlflow:
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "predictive-healthcare-meds"))
            mlflow.start_run()

            # Log parameters
            mlflow.log_params({
                "model_type": type(self.model).__name__,
                "learning_rate": self.train_config.get("learning_rate"),
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": self.model_config.get("optimizer", {}).get("type", "Adam"),
            })

        logger.info(f"Starting training for {epochs} epochs "
                     f"(train: {len(X_train)}, val: {len(X_val)} samples)")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # ── Training ──────────────────────────────────────────────────────
            train_loss, train_acc = self._train_epoch(train_loader)

            # ── Validation ────────────────────────────────────────────────────
            val_loss, val_acc = self._validate_epoch(val_loader)

            # ── Record history ────────────────────────────────────────────────
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # ── Learning rate scheduling ──────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ── Logging ───────────────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e}"
            )

            # ── MLflow logging ───────────────────────────────────────────────
            if use_mlflow:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                }, step=epoch)

            # ── Early stopping ────────────────────────────────────────────────
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                self.early_stopping.restore_best(self.model)
                break

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s "
                     f"({epoch} epochs, best val_loss={self.early_stopping.best_score:.4f})")

        # Save final model checkpoint
        self._save_checkpoint("last")

        # MLflow: log final model
        if use_mlflow:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()

        return history

    def _train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).float()

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(batch_x).squeeze(-1)

            # Compute loss
            loss = self.criterion(logits, batch_y)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).float()

            logits = self.model(batch_x).squeeze(-1)
            loss = self.criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from numpy arrays.

        Handles both 2D (tabular for MLP) and 3D (sequence for LSTM/Transformer)
        input shapes automatically.
        """
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        dataset = TensorDataset(X_tensor, y_tensor)
        num_workers = self.global_config.get("num_workers", 0)
        pin_memory = self.global_config.get("pin_memory", False)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def _save_checkpoint(self, tag: str = "best") -> None:
        """Save a model checkpoint to disk."""
        ckpt_cfg = self.model_config.get("checkpoint", {})
        dirpath = Path(ckpt_cfg.get("dirpath", "results/models/checkpoints"))
        dirpath.mkdir(parents=True, exist_ok=True)

        model_name = type(self.model).__name__
        filepath = dirpath / f"{model_name}_{tag}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_type": model_name,
        }, filepath)

        logger.info(f"Saved {tag} checkpoint to {filepath}")
