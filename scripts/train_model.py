#!/usr/bin/env python
"""
train_model.py - Full training pipeline entry point.

Orchestrates the complete end-to-end training workflow:
1. Load configuration
2. Load/generate data
3. Preprocess data
4. Engineer features
5. Split into train/val/test
6. Build model
7. Train with early stopping
8. Evaluate on test set
9. Save model and results

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --model lstm
    python scripts/train_model.py --config configs/model_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.preprocess import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.model import ModelFactory
from src.models.train import Trainer
from src.evaluation.evaluate import Evaluator
from src.utils.config_loader import load_config, merge_configs
from src.utils.logger import get_logger, setup_logging_from_config

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a predictive healthcare model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mlp", "lstm", "transformer"],
        help="Model type to train (overrides config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data file (overrides auto-discovery)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.

    Uses stratified splitting to maintain class balance across splits,
    which is critical for imbalanced healthcare outcomes.

    Args:
        X: Feature array.
        y: Label array.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        stratify: Whether to stratify splits by label.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y if stratify else None,
        random_state=42,
    )

    # Second split: train vs val
    val_fraction = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction,
        stratify=y_train_val if stratify else None,
        random_state=42,
    )

    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    # Load configurations
    global_config = load_config("configs/config.yaml")
    model_config = load_config(args.config)
    data_config = load_config("configs/data_config.yaml")

    # Setup logging
    setup_logging_from_config(global_config)
    logger.info("=" * 60)
    logger.info("PREDICTIVE HEALTHCARE - Training Pipeline")
    logger.info("=" * 60)

    # Set seed
    set_seed(args.seed)

    # Apply CLI overrides
    if args.model:
        model_config["active_model"] = args.model
    if args.epochs:
        model_config["training"]["epochs"] = args.epochs
    if args.batch_size:
        model_config["training"]["batch_size"] = args.batch_size
    if args.lr:
        model_config["training"]["learning_rate"] = args.lr

    model_type = model_config.get("active_model", "mlp")
    logger.info(f"Active model: {model_type}")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    logger.info("Step 1: Loading data")
    loader = DataLoader(data_config)
    df = loader.load(args.data_path)
    logger.info(f"Data shape: {df.shape}")

    # Save raw data if generated
    if args.data_path is None:
        loader.save(df, "data/raw/healthcare_data.csv")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    logger.info("Step 2: Preprocessing data")
    preprocessor = Preprocessor(data_config.get("preprocessing", {}))
    columns_config = data_config.get("columns", {})
    df_processed = preprocessor.fit_transform(df, columns_config)

    # ── Step 3: Feature engineering ──────────────────────────────────────────
    logger.info("Step 3: Engineering features")
    engineer = FeatureEngineer(data_config.get("feature_engineering", {}))
    df_engineered = engineer.transform(df_processed)

    # ── Step 4: Prepare model inputs ─────────────────────────────────────────
    logger.info("Step 4: Preparing model inputs")
    target_col = data_config.get("raw_data", {}).get("target_column", "readmission_30day")
    feature_cols = engineer.get_feature_names(df_engineered, exclude_cols=[target_col])

    # Determine if we need sequence data (for LSTM/Transformer)
    if model_type in ("lstm", "transformer"):
        # Time-series preprocessing
        df_ts = preprocessor.preprocess_time_series(df_processed)
        feature_cols_ts = [c for c in feature_cols if c in df_ts.columns and c in df_ts.select_dtypes(include=[np.number]).columns]
        ts_cfg = data_config.get("preprocessing", {}).get("time_series", {})
        X, y = preprocessor.create_sequences(
            df_ts, feature_cols_ts, target_col,
            lookback=ts_cfg.get("lookback_window", 48),
            horizon=ts_cfg.get("forecast_horizon", 24),
        )
    else:
        # Tabular data for MLP
        available_features = [c for c in feature_cols if c in df_engineered.columns]
        X = df_engineered[available_features].values.astype(np.float32)
        y = df_engineered[target_col].values.astype(np.float32)

        # Fill any remaining NaN with 0
        X = np.nan_to_num(X, nan=0.0)

    # Update model input_dim to match actual feature count
    if model_type == "mlp":
        model_config["mlp"]["input_dim"] = X.shape[-1]
    else:
        model_config[model_type]["input_dim"] = X.shape[-1]

    # ── Step 5: Split data ───────────────────────────────────────────────────
    logger.info("Step 5: Splitting data")
    split_cfg = global_config.get("data_split", {})
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        stratify=split_cfg.get("stratify", True),
    )

    # ── Step 6: Create model ─────────────────────────────────────────────────
    logger.info("Step 6: Creating model")
    model = ModelFactory.from_config(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # ── Step 7: Train ────────────────────────────────────────────────────────
    logger.info("Step 7: Training model")
    trainer = Trainer(model, model_config, global_config)
    history = trainer.train(X_train, y_train, X_val, y_val)

    # ── Step 8: Evaluate ─────────────────────────────────────────────────────
    logger.info("Step 8: Evaluating model")
    evaluator = Evaluator(predictor=trainer.predictor if hasattr(trainer, 'predictor') else None,
                          model=model,
                          results_dir="results/metrics")
    # Create predictor from trained model for evaluation
    from src.models.predict import Predictor
    predictor = Predictor(model=model, model_config=model_config)
    evaluator = Evaluator(predictor=predictor, results_dir="results/metrics")
    results = evaluator.evaluate(X_test, y_test, model_name=model_type)

    # ── Step 9: Save model ───────────────────────────────────────────────────
    logger.info("Step 9: Saving model")
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save_model(model_dir / f"{model_type}_final.pt")

    # Save training history
    history_path = model_dir / f"{model_type}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Test Accuracy: {results['default_metrics']['accuracy']:.4f}")
    logger.info(f"Test F1: {results['default_metrics']['f1_score']:.4f}")
    logger.info(f"Test AUC-ROC: {results['default_metrics'].get('auc_roc', 'N/A')}")
    logger.info(f"Optimal Threshold (F1): {results['summary']['optimal_threshold_by_f1']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
