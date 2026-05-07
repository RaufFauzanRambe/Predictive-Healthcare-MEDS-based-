#!/usr/bin/env python
"""
run_inference.py - Run predictions using a trained model.

Loads a trained model checkpoint and runs inference on new data.
Supports both batch predictions from files and single-patient
predictions via command-line arguments.

Usage:
    python scripts/run_inference.py --checkpoint results/models/lstm_final.pt
    python scripts/run_inference.py --checkpoint results/models/mlp_best.pt --data data/raw/new_data.csv
    python scripts/run_inference.py --checkpoint results/models/transformer_final.pt --output results/predictions/preds.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.preprocess import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.predict import Predictor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, setup_logging_from_config

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to input data file (CSV/JSON/Parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions/predictions.json",
        help="Path to save prediction results",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["mlp", "lstm", "transformer"],
        help="Model type (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for positive prediction",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Run inference for a specific patient ID only",
    )
    return parser.parse_args()


def main() -> None:
    """Run the inference pipeline."""
    args = parse_args()

    # Load configurations
    global_config = load_config("configs/config.yaml")
    model_config = load_config("configs/model_config.yaml")
    data_config = load_config("configs/data_config.yaml")

    # Setup logging
    setup_logging_from_config(global_config)
    logger.info("=" * 60)
    logger.info("PREDICTIVE HEALTHCARE - Inference Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load model ───────────────────────────────────────────────────
    logger.info(f"Loading model from {args.checkpoint}")
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        model_config=model_config,
    )
    predictor.threshold = args.threshold

    # ── Step 2: Load and preprocess data ─────────────────────────────────────
    logger.info("Loading data for inference")
    loader = DataLoader(data_config)
    df = loader.load(args.data)
    logger.info(f"Loaded {len(df)} records")

    # Filter for specific patient if requested
    if args.patient_id:
        df = df[df["patient_id"] == args.patient_id]
        if df.empty:
            logger.error(f"Patient {args.patient_id} not found in data")
            sys.exit(1)
        logger.info(f"Filtered to patient {args.patient_id}: {len(df)} records")

    # Preprocess
    preprocessor = Preprocessor(data_config.get("preprocessing", {}))
    columns_config = data_config.get("columns", {})
    df_processed = preprocessor.fit_transform(df, columns_config)

    # Feature engineering
    engineer = FeatureEngineer(data_config.get("feature_engineering", {}))
    df_engineered = engineer.transform(df_processed)

    # Prepare features
    target_col = data_config.get("raw_data", {}).get("target_column", "readmission_30day")
    feature_cols = engineer.get_feature_names(df_engineered, exclude_cols=[target_col])
    available_features = [c for c in feature_cols if c in df_engineered.columns]

    X = df_engineered[available_features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    # ── Step 3: Run predictions ──────────────────────────────────────────────
    logger.info("Running predictions")
    results = predictor.predict(X, return_probs=True)

    # ── Step 4: Format and save results ──────────────────────────────────────
    output_records = []
    for i in range(len(results["predictions"])):
        record = {
            "sample_index": i,
            "prediction": int(results["predictions"][i]),
            "probability": float(results["probabilities"][i]),
            "risk_level": predictor._classify_risk(results["probabilities"][i]),
        }
        # Add patient ID if available
        if "patient_id" in df.columns and i < len(df):
            record["patient_id"] = str(df.iloc[i].get("patient_id", ""))
        output_records.append(record)

    # Summary statistics
    summary = {
        "total_samples": len(results["predictions"]),
        "predicted_positive": int(results["predictions"].sum()),
        "predicted_negative": int(len(results["predictions"]) - results["predictions"].sum()),
        "positive_rate": float(results["predictions"].mean()),
        "mean_probability": float(results["probabilities"].mean()),
        "max_probability": float(results["probabilities"].max()),
        "min_probability": float(results["probabilities"].min()),
        "threshold_used": args.threshold,
        "checkpoint_used": args.checkpoint,
    }

    output = {
        "summary": summary,
        "predictions": output_records,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Predicted positive: {summary['predicted_positive']} ({summary['positive_rate']:.1%})")
    logger.info(f"Mean probability: {summary['mean_probability']:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
