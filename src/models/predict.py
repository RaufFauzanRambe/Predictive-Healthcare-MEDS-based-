"""
predict.py - Model inference and prediction utilities.

Provides a Predictor class for loading trained models and running
inference on new data, with support for batch predictions and
probability calibration.
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.model import ModelFactory
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class Predictor:
    """
    Model inference pipeline for healthcare predictions.

    Handles model loading from checkpoints, data preprocessing for inference,
    and producing predictions with calibrated probabilities.

    Attributes:
        model: Loaded PyTorch model.
        device: Torch device for computation.
        threshold: Classification threshold (default 0.5).
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_config: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Predictor.

        Either a model instance or a checkpoint_path must be provided.

        Args:
            model: Pre-loaded PyTorch model. Takes precedence over checkpoint.
            checkpoint_path: Path to a saved model checkpoint.
            model_config: Model configuration for reconstruction.
            device: Device string. Auto-detects if None.
        """
        # Device setup
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        if model_config is None:
            model_config = load_config("configs/model_config.yaml")

        self.model_config = model_config
        self.threshold = 0.5

        # Load model
        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError("Either 'model' or 'checkpoint_path' must be provided")

        self.model.eval()
        logger.info(f"Predictor initialized on {self.device}")

    def _load_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> nn.Module:
        """
        Load a model from a saved checkpoint file.

        The checkpoint should contain 'model_state_dict' and 'model_type' keys
        as saved by the Trainer._save_checkpoint method.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.

        Returns:
            Loaded and initialized PyTorch model.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model_type = checkpoint.get("model_type", "mlp")
        model_params = self.model_config.get(model_type, {})
        model = ModelFactory.create(model_type, **model_params)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        return model

    @torch.no_grad()
    def predict(self, X: np.ndarray, return_probs: bool = True) -> dict[str, np.ndarray]:
        """
        Run inference on input data.

        Processes input through the model and returns both class predictions
        and probability scores. For binary classification, probabilities are
        for the positive class (readmission).

        Args:
            X: Input features. Shape depends on model type:
               - MLP: (n_samples, n_features)
               - LSTM/Transformer: (n_samples, seq_len, n_features)
            return_probs: Whether to return calibrated probabilities.

        Returns:
            Dictionary with:
            - 'predictions': Binary class predictions (0 or 1)
            - 'probabilities': Positive class probabilities (if return_probs=True)
            - 'logits': Raw model logits
        """
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # Ensure correct dimensionality for MLP (2D) vs sequence models (3D)
        if len(X_tensor.shape) == 2 and hasattr(self.model, 'lstm'):
            # Add sequence dimension for LSTM/Transformer
            X_tensor = X_tensor.unsqueeze(1)

        logits = self.model(X_tensor).squeeze(-1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= self.threshold).astype(int)
        logits_np = logits.cpu().numpy()

        result = {
            "predictions": predictions,
            "logits": logits_np,
        }
        if return_probs:
            result["probabilities"] = probabilities

        logger.info(f"Predictions: {len(predictions)} samples, "
                     f"positive rate={predictions.mean():.3f}")
        return result

    def predict_single(self, x: np.ndarray) -> dict[str, Any]:
        """
        Run inference on a single sample.

        Convenience method that wraps predict() for single-sample inputs.

        Args:
            x: Single input sample.

        Returns:
            Dictionary with prediction details.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 2 and hasattr(self.model, 'lstm'):
            x = x.reshape(1, x.shape[0], x.shape[1])

        result = self.predict(x, return_probs=True)
        return {
            "prediction": int(result["predictions"][0]),
            "probability": float(result["probabilities"][0]),
            "logit": float(result["logits"][0]),
            "risk_level": self._classify_risk(result["probabilities"][0]),
        }

    @staticmethod
    def _classify_risk(prob: float) -> str:
        """
        Classify the risk level based on predicted probability.

        Args:
            prob: Predicted probability of positive outcome.

        Returns:
            Risk level string: 'low', 'moderate', or 'high'.
        """
        if prob < 0.3:
            return "low"
        elif prob < 0.6:
            return "moderate"
        else:
            return "high"

    def save_model(self, output_path: Union[str, Path]) -> None:
        """
        Save the current model to a file.

        Args:
            output_path: Path to save the model checkpoint.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": type(self.model).__name__,
            "threshold": self.threshold,
        }, output_path)

        logger.info(f"Model saved to {output_path}")

    def load_model(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a model from a checkpoint, replacing the current model.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        self.model = self._load_from_checkpoint(checkpoint_path)
        self.model.eval()
        logger.info("Model loaded and set to eval mode")
