"""
experiment_runner.py - Engine utama untuk menjalankan eksperimen ML.

Menyediakan ExperimentRunner yang menangani:
- Eksperimen model tunggal dengan konfigurasi kustom
- Perbandingan multi-model (MLP vs LSTM vs Transformer)
- Hyperparameter grid search
- MLflow experiment tracking otomatis
- Penyimpanan dan pemuatan hasil eksperimen
- Logging hasil ke file JSON untuk analisis lebih lanjut

Usage:
    from experiments.experiment_runner import ExperimentRunner

    runner = ExperimentRunner()
    results = runner.run_single(model_type="lstm", overrides={"training": {"learning_rate": 0.0005}})
    comparison = runner.run_model_comparison()
    sweep_results = runner.run_hyperparameter_sweep("lstm", param_grid)
"""

import copy
import json
import time
import uuid
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch

from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.preprocess import Preprocessor
from src.evaluation.evaluate import Evaluator
from src.evaluation.metrics import compute_metrics
from src.features.feature_engineering import FeatureEngineer
from src.models.model import ModelFactory
from src.models.predict import Predictor
from src.models.train import Trainer
from src.utils.config_loader import load_config, merge_configs
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentRunner:
    """
    Engine eksperimen untuk predictive healthcare modeling.

    Mengorkestrasi keseluruhan pipeline eksperimen: memuat data,
    preprocessing, feature engineering, training, evaluasi, dan
    penyimpanan hasil. Terintegrasi penuh dengan MLflow untuk
    tracking parameter, metric, dan artifact.

    Attributes:
        global_config: Konfigurasi global dari config.yaml.
        model_config: Konfigurasi model dari model_config.yaml.
        data_config: Konfigurasi data dari data_config.yaml.
        experiment_dir: Direktori untuk menyimpan hasil eksperimen.
    """

    def __init__(
        self,
        global_config_path: str = "configs/config.yaml",
        model_config_path: str = "configs/model_config.yaml",
        data_config_path: str = "configs/data_config.yaml",
        experiment_dir: str = "experiments",
    ) -> None:
        """
        Inisialisasi ExperimentRunner.

        Args:
            global_config_path: Path ke konfigurasi global.
            model_config_path: Path ke konfigurasi model.
            data_config_path: Path ke konfigurasi data.
            experiment_dir: Direktori output untuk hasil eksperimen.
        """
        self.global_config = load_config(global_config_path)
        self.model_config = load_config(model_config_path)
        self.data_config = load_config(data_config_path)
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Cache data yang sudah diproses (hindari reprocessing berulang)
        self._cached_data: Optional[dict[str, np.ndarray]] = None
        self._preprocessor: Optional[Preprocessor] = None

        # Setup MLflow
        mlflow_cfg = self.global_config.get("mlflow", {})
        if mlflow_cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

        logger.info("ExperimentRunner diinisialisasi")

    # =========================================================================
    # Data Loading & Preprocessing (shared across experiments)
    # =========================================================================

    def prepare_data(
        self,
        force_reload: bool = False,
        model_type: str = "mlp",
    ) -> dict[str, np.ndarray]:
        """
        Muat dan preprocess data, cache untuk penggunaan berulang.

        Seluruh eksperimen dalam satu sesi menggunakan data yang sama
        untuk memastikan perbandingan yang adil. Data hanya di-preprocess
        sekali dan di-cache di memori.

        Args:
            force_reload: Paksa muat ulang data dari disk.
            model_type: Tipe model ('mlp', 'lstm', 'transformer') —
                        menentukan format input (tabular vs sequence).

        Returns:
            Dictionary dengan keys: X_train, y_train, X_val, y_val,
            X_test, y_test.
        """
        if self._cached_data is not None and not force_reload:
            return self._cached_data

        logger.info("Memuat dan memproses data...")

        # Set seed untuk reproducibility
        seed = self.global_config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Step 1: Load data
        loader = DataLoader(self.data_config)
        df = loader.load()

        # Step 2: Preprocess
        if self._preprocessor is None:
            self._preprocessor = Preprocessor(self.data_config.get("preprocessing", {}))
        columns_config = self.data_config.get("columns", {})
        df_processed = self._preprocessor.fit_transform(df, columns_config)

        # Step 3: Feature engineering
        engineer = FeatureEngineer(self.data_config.get("feature_engineering", {}))
        df_engineered = engineer.transform(df_processed)

        # Step 4: Prepare model-specific inputs
        target_col = self.data_config.get("raw_data", {}).get(
            "target_column", "readmission_30day"
        )
        feature_cols = engineer.get_feature_names(
            df_engineered, exclude_cols=[target_col]
        )

        if model_type in ("lstm", "transformer"):
            # Time-series data untuk model sekuensial
            df_ts = self._preprocessor.preprocess_time_series(df_processed)
            feature_cols_ts = [
                c for c in feature_cols
                if c in df_ts.columns
                and c in df_ts.select_dtypes(include=[np.number]).columns
            ]
            ts_cfg = self.data_config.get("preprocessing", {}).get("time_series", {})
            X, y = self._preprocessor.create_sequences(
                df_ts, feature_cols_ts, target_col,
                lookback=ts_cfg.get("lookback_window", 48),
                horizon=ts_cfg.get("forecast_horizon", 24),
            )
        else:
            # Tabular data untuk MLP
            available_features = [
                c for c in feature_cols if c in df_engineered.columns
            ]
            X = df_engineered[available_features].values.astype(np.float32)
            y = df_engineered[target_col].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)

        # Step 5: Split data
        from sklearn.model_selection import train_test_split

        split_cfg = self.global_config.get("data_split", {})
        train_ratio = split_cfg.get("train_ratio", 0.7)
        val_ratio = split_cfg.get("val_ratio", 0.15)
        stratify = split_cfg.get("stratify", True)
        test_ratio = 1.0 - train_ratio - val_ratio

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_ratio,
            stratify=y if stratify else None, random_state=seed,
        )
        val_fraction = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_fraction,
            stratify=y_train_val if stratify else None, random_state=seed,
        )

        self._cached_data = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        }

        # Update input_dim berdasarkan data aktual
        self._actual_input_dim = X.shape[-1]

        logger.info(
            f"Data siap: train={len(X_train)}, val={len(X_val)}, "
            f"test={len(X_test)}, features={X.shape[-1]}"
        )
        return self._cached_data

    # =========================================================================
    # Single Experiment Run
    # =========================================================================

    def run_single(
        self,
        model_type: str = "lstm",
        overrides: Optional[dict] = None,
        run_name: Optional[str] = None,
        save_model: bool = True,
    ) -> dict[str, Any]:
        """
        Jalankan satu eksperimen training lengkap.

        Melakukan training model dengan konfigurasi yang diberikan,
        evaluasi pada test set, dan mencatat semua hasil ke MLflow
        serta file JSON lokal.

        Args:
            model_type: Tipe model ('mlp', 'lstm', 'transformer').
            overrides: Dictionary override untuk model_config. Mendukung
                       nested keys, misal {"training": {"learning_rate": 0.0005}}.
            run_name: Nama unik untuk run ini. Auto-generated jika None.
            save_model: Apakah menyimpan model ke disk.

        Returns:
            Dictionary berisi semua hasil eksperimen: konfigurasi,
            metrik, history, dan metadata.
        """
        if run_name is None:
            run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("=" * 70)
        logger.info(f"MULAI EKSPERIMEN: {run_name}")
        logger.info(f"Model: {model_type}")
        logger.info("=" * 70)

        # Gabungkan konfigurasi dengan override
        config = copy.deepcopy(self.model_config)
        if overrides:
            config = merge_configs(config, overrides)
        config["active_model"] = model_type

        # Siapkan data
        data = self.prepare_data(model_type=model_type)

        # Update input_dim berdasarkan data aktual
        config[model_type]["input_dim"] = self._actual_input_dim

        # Buat model
        model = ModelFactory.from_config(config)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameter model: {param_count:,}")

        # Training
        start_time = time.time()
        trainer = Trainer(model, config, self.global_config)
        history = trainer.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
        )
        training_time = time.time() - start_time

        # Evaluasi pada test set
        predictor = Predictor(model=model, model_config=config)
        eval_results = self._evaluate_model(predictor, data["X_test"], data["y_test"])

        # Simpan model
        if save_model:
            model_dir = self.experiment_dir / "saved_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{run_name}.pt"
            predictor.save_model(model_path)
            eval_results["model_path"] = str(model_path)

        # Kompilasi hasil eksperimen
        results = {
            "run_id": str(uuid.uuid4())[:8],
            "run_name": run_name,
            "model_type": model_type,
            "parameter_count": param_count,
            "training_time_seconds": round(training_time, 2),
            "config": config,
            "overrides": overrides or {},
            "history": history,
            "metrics": eval_results["default_metrics"],
            "threshold_analysis": eval_results["threshold_analysis"],
            "summary": eval_results["summary"],
            "timestamp": datetime.now().isoformat(),
        }

        # Simpan hasil ke JSON
        self._save_results(results, run_name)

        logger.info("=" * 70)
        logger.info(f"EKSPERIMEN SELESAI: {run_name}")
        logger.info(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        logger.info(f"  F1 Score: {results['metrics']['f1_score']:.4f}")
        logger.info(f"  AUC-ROC:  {results['metrics'].get('auc_roc', 'N/A')}")
        logger.info(f"  Waktu:    {training_time:.1f}s")
        logger.info("=" * 70)

        return results

    # =========================================================================
    # Model Comparison
    # =========================================================================

    def run_model_comparison(
        self,
        model_types: Optional[list[str]] = None,
        overrides_per_model: Optional[dict[str, dict]] = None,
    ) -> dict[str, Any]:
        """
        Bandingkan performa beberapa arsitektur model.

        Menjalankan training dan evaluasi untuk setiap tipe model
        dengan data yang identik, menghasilkan tabel perbandingan
        dan menentukan model terbaik berdasarkan F1 score.

        Args:
            model_types: Daftar tipe model untuk dibandingkan.
                         Default: ['mlp', 'lstm', 'transformer'].
            overrides_per_model: Override konfigurasi per tipe model.

        Returns:
            Dictionary berisi hasil perbandingan: individual results,
            comparison table, dan best model info.
        """
        if model_types is None:
            model_types = ["mlp", "lstm", "transformer"]

        if overrides_per_model is None:
            overrides_per_model = {}

        logger.info("=" * 70)
        logger.info("PERBANDINGAN MULTI-MODEL")
        logger.info(f"Model: {model_types}")
        logger.info("=" * 70)

        all_results = {}
        for model_type in model_types:
            overrides = overrides_per_model.get(model_type, {})
            result = self.run_single(
                model_type=model_type,
                overrides=overrides,
                run_name=f"comparison_{model_type}",
                save_model=True,
            )
            all_results[model_type] = result

        # Buat tabel perbandingan
        comparison = self._build_comparison_table(all_results)

        # Tentukan model terbaik
        best_model = max(
            all_results.items(),
            key=lambda x: x[1]["metrics"]["f1_score"],
        )

        summary = {
            "experiment_type": "model_comparison",
            "model_types": model_types,
            "best_model": best_model[0],
            "best_f1_score": best_model[1]["metrics"]["f1_score"],
            "best_accuracy": best_model[1]["metrics"]["accuracy"],
            "best_auc_roc": best_model[1]["metrics"].get("auc_roc"),
            "comparison_table": comparison,
            "timestamp": datetime.now().isoformat(),
        }

        # Simpan perbandingan
        comp_path = self.experiment_dir / "model_comparison.json"
        with open(comp_path, "w") as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)

        logger.info("\n" + "=" * 70)
        logger.info("HASIL PERBANDINGAN")
        logger.info("-" * 70)
        for row in comparison:
            logger.info(
                f"  {row['model_type']:15s} | "
                f"Acc={row['accuracy']:.4f} | "
                f"F1={row['f1_score']:.4f} | "
                f"AUC={row.get('auc_roc', 'N/A')} | "
                f"Params={row['param_count']:>8,} | "
                f"Time={row['training_time']:.1f}s"
            )
        logger.info("-" * 70)
        logger.info(f"  MODEL TERBAIK: {best_model[0]} (F1={best_model[1]['metrics']['f1_score']:.4f})")
        logger.info("=" * 70)

        return {"individual_results": all_results, "summary": summary}

    # =========================================================================
    # Hyperparameter Sweep
    # =========================================================================

    def run_hyperparameter_sweep(
        self,
        model_type: str = "lstm",
        param_grid: Optional[dict[str, list]] = None,
        metric_to_optimize: str = "f1_score",
    ) -> dict[str, Any]:
        """
        Jalankan grid search hyperparameter untuk satu arsitektur model.

        Mencoba semua kombinasi parameter dari param_grid, melacak
        setiap run di MLflow, dan mengidentifikasi konfigurasi terbaik.
        Mendukung nested parameter keys menggunakan notasi dot
        (misalnya "training.learning_rate").

        Args:
            model_type: Tipe model untuk di-sweep.
            param_grid: Dictionary parameter → daftar nilai.
                        Contoh: {"training.learning_rate": [0.001, 0.0005],
                                  "training.batch_size": [32, 64]}
            metric_to_optimize: Metrik untuk menentukan konfigurasi terbaik.

        Returns:
            Dictionary berisi semua hasil sweep dan konfigurasi terbaik.
        """
        if param_grid is None:
            param_grid = {
                "training.learning_rate": [0.001, 0.0005, 0.0001],
                "training.batch_size": [32, 64],
                "training.epochs": [30, 50],
            }

        # Generate semua kombinasi parameter
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info("=" * 70)
        logger.info(f"HYPERPARAMETER SWEEP: {model_type}")
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Total kombinasi: {len(combinations)}")
        logger.info(f"Metrik optimasi: {metric_to_optimize}")
        logger.info("=" * 70)

        sweep_results = []
        best_score = -float("inf")
        best_config = None

        for i, combo in enumerate(combinations, 1):
            # Konversi flat keys ke nested dict
            overrides = self._flat_to_nested(dict(zip(param_names, combo)))

            run_name = f"sweep_{model_type}_run{i:03d}"
            logger.info(f"\n--- Sweep Run {i}/{len(combinations)}: {dict(zip(param_names, combo))} ---")

            try:
                result = self.run_single(
                    model_type=model_type,
                    overrides=overrides,
                    run_name=run_name,
                    save_model=False,  # Jangan simpan semua model saat sweep
                )

                score = result["metrics"].get(metric_to_optimize, 0)
                sweep_entry = {
                    "run_index": i,
                    "run_name": run_name,
                    "params": dict(zip(param_names, [str(v) for v in combo])),
                    "score": score,
                    "metrics": result["metrics"],
                    "training_time": result["training_time_seconds"],
                }
                sweep_results.append(sweep_entry)

                if score > best_score:
                    best_score = score
                    best_config = dict(zip(param_names, combo))
                    logger.info(f"  >>> Best {metric_to_optimize} baru: {score:.4f}")

            except Exception as e:
                logger.error(f"  Sweep run {i} gagal: {e}")
                sweep_results.append({
                    "run_index": i,
                    "run_name": run_name,
                    "params": dict(zip(param_names, [str(v) for v in combo])),
                    "error": str(e),
                })

        # Ringkasan sweep
        summary = {
            "experiment_type": "hyperparameter_sweep",
            "model_type": model_type,
            "param_grid": {k: [str(v) for v in vals] for k, vals in param_grid.items()},
            "total_runs": len(combinations),
            "successful_runs": len([r for r in sweep_results if "error" not in r]),
            "metric_to_optimize": metric_to_optimize,
            "best_config": {k: str(v) for k, v in best_config.items()} if best_config else None,
            "best_score": best_score,
            "all_results": sweep_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Simpan hasil sweep
        sweep_path = self.experiment_dir / f"sweep_{model_type}_results.json"
        with open(sweep_path, "w") as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)

        logger.info("\n" + "=" * 70)
        logger.info("HASIL HYPERPARAMETER SWEEP")
        logger.info("-" * 70)
        for r in sweep_results:
            if "error" not in r:
                logger.info(
                    f"  Run {r['run_index']:3d} | "
                    f"{metric_to_optimize}={r['score']:.4f} | "
                    f"params={r['params']}"
                )
        logger.info("-" * 70)
        if best_config:
            logger.info(f"  KONFIGURASI TERBAIK: {best_config}")
            logger.info(f"  {metric_to_optimize} = {best_score:.4f}")
        logger.info("=" * 70)

        return summary

    # =========================================================================
    # Ablation Study
    # =========================================================================

    def run_ablation_study(
        self,
        model_type: str = "lstm",
        ablation_config: Optional[dict[str, dict]] = None,
    ) -> dict[str, Any]:
        """
        Jalankan ablation study untuk menganalisis kontribusi setiap komponen.

        Membandingkan performa model penuh dengan model yang memiliki
        komponen yang dihilangkan satu per satu. Berguna untuk memahami
        komponen mana yang paling berkontribusi terhadap performa.

        Args:
            model_type: Tipe model dasar.
            ablation_config: Dictionary nama_ablasi → override config.
                Contoh: {
                    "no_attention": {"lstm": {"attention": False}},
                    "unidirectional": {"lstm": {"bidirectional": False}},
                    "single_layer": {"lstm": {"num_layers": 1}},
                    "no_dropout": {"training": {"dropout": 0.0}},
                }

        Returns:
            Dictionary berisi hasil ablation study.
        """
        if ablation_config is None:
            if model_type == "lstm":
                ablation_config = {
                    "full_model": {},
                    "no_attention": {"lstm": {"attention": False}},
                    "unidirectional": {"lstm": {"bidirectional": False}},
                    "single_layer": {"lstm": {"num_layers": 1}},
                    "smaller_hidden": {"lstm": {"hidden_dim": 64}},
                }
            elif model_type == "transformer":
                ablation_config = {
                    "full_model": {},
                    "single_layer": {"transformer": {"num_encoder_layers": 1}},
                    "smaller_dmodel": {"transformer": {"d_model": 64}},
                    "no_cls_learned_pe": {"transformer": {"positional_encoding": "learned"}},
                    "fewer_heads": {"transformer": {"nhead": 2}},
                }
            else:  # mlp
                ablation_config = {
                    "full_model": {},
                    "narrower": {"mlp": {"hidden_dims": [128, 64]}},
                    "deeper": {"mlp": {"hidden_dims": [256, 128, 64, 32]}},
                    "no_batchnorm": {"mlp": {"use_batch_norm": False}},
                }

        logger.info("=" * 70)
        logger.info(f"ABLATION STUDY: {model_type}")
        logger.info(f"Variants: {list(ablation_config.keys())}")
        logger.info("=" * 70)

        ablation_results = {}
        for name, overrides in ablation_config.items():
            logger.info(f"\n--- Ablation: {name} ---")
            result = self.run_single(
                model_type=model_type,
                overrides=overrides,
                run_name=f"ablation_{model_type}_{name}",
                save_model=False,
            )
            ablation_results[name] = {
                "overrides": overrides,
                "metrics": result["metrics"],
                "training_time": result["training_time_seconds"],
                "parameter_count": result["parameter_count"],
            }

        # Bandingkan dengan full model
        full_metrics = ablation_results.get("full_model", {}).get("metrics", {})
        summary = {
            "experiment_type": "ablation_study",
            "model_type": model_type,
            "full_model_f1": full_metrics.get("f1_score"),
            "full_model_accuracy": full_metrics.get("accuracy"),
            "ablation_results": ablation_results,
            "impact_ranking": self._rank_ablation_impact(ablation_results),
            "timestamp": datetime.now().isoformat(),
        }

        # Simpan
        abl_path = self.experiment_dir / f"ablation_{model_type}_results.json"
        with open(abl_path, "w") as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)

        logger.info("\n" + "=" * 70)
        logger.info("HASIL ABLATION STUDY")
        logger.info("-" * 70)
        for name, res in ablation_results.items():
            delta_f1 = res["metrics"]["f1_score"] - full_metrics.get("f1_score", 0)
            logger.info(
                f"  {name:25s} | "
                f"F1={res['metrics']['f1_score']:.4f} | "
                f"Delta={delta_f1:+.4f} | "
                f"Params={res['parameter_count']:>8,}"
            )
        logger.info("=" * 70)

        return summary

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _evaluate_model(
        self, predictor: Predictor, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, Any]:
        """Jalankan evaluasi model pada test set."""
        evaluator = Evaluator(predictor=predictor, results_dir="results/metrics")
        return evaluator.evaluate(X_test, y_test, model_name="experiment", save_results=False)

    def _save_results(self, results: dict, run_name: str) -> None:
        """Simpan hasil eksperimen ke file JSON."""
        results_dir = self.experiment_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / f"{run_name}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=self._json_serializer)

        logger.info(f"Hasil disimpan ke {filepath}")

    @staticmethod
    def _build_comparison_table(all_results: dict) -> list[dict]:
        """Buat tabel perbandingan dari hasil multi-model."""
        table = []
        for model_type, result in all_results.items():
            metrics = result["metrics"]
            table.append({
                "model_type": model_type,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "auc_roc": metrics.get("auc_roc", 0),
                "specificity": metrics.get("specificity", 0),
                "balanced_accuracy": metrics.get("balanced_accuracy", 0),
                "param_count": result.get("parameter_count", 0),
                "training_time": result.get("training_time_seconds", 0),
            })
        return table

    @staticmethod
    def _rank_ablation_impact(ablation_results: dict) -> list[dict]:
        """Ranking dampak setiap komponen berdasarkan perubahan F1."""
        full_f1 = ablation_results.get("full_model", {}).get("metrics", {}).get("f1_score", 0)
        impacts = []

        for name, res in ablation_results.items():
            if name == "full_model":
                continue
            f1 = res["metrics"]["f1_score"]
            delta = f1 - full_f1
            impacts.append({
                "component": name,
                "f1_score": f1,
                "delta_f1": round(delta, 4),
                "impact": "critical" if delta < -0.05 else
                          "significant" if delta < -0.02 else
                          "minor" if delta < 0 else
                          "neutral",
            })

        impacts.sort(key=lambda x: x["delta_f1"])
        return impacts

    @staticmethod
    def _flat_to_nested(flat_dict: dict[str, Any]) -> dict:
        """
        Konversi dictionary flat dengan dot-notation ke nested dict.

        Contoh: {"training.learning_rate": 0.001}
                → {"training": {"learning_rate": 0.001}}
        """
        nested: dict = {}
        for key, value in flat_dict.items():
            parts = key.split(".")
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return nested

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Serializer kustom untuk JSON (handle numpy/torch types)."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def load_results(self, run_name: str) -> Optional[dict]:
        """
        Muat hasil eksperimen dari file JSON.

        Args:
            run_name: Nama run yang disimpan sebelumnya.

        Returns:
            Dictionary hasil eksperimen, atau None jika tidak ditemukan.
        """
        filepath = self.experiment_dir / "results" / f"{run_name}.json"
        if not filepath.exists():
            logger.warning(f"Hasil eksperimen tidak ditemukan: {filepath}")
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def list_experiments(self) -> list[dict]:
        """
        List semua eksperimen yang tersimpan.

        Returns:
            List dictionary berisi metadata setiap eksperimen.
        """
        results_dir = self.experiment_dir / "results"
        if not results_dir.exists():
            return []

        experiments = []
        for filepath in results_dir.glob("*.json"):
            with open(filepath, "r") as f:
                data = json.load(f)
            experiments.append({
                "run_name": data.get("run_name", filepath.stem),
                "model_type": data.get("model_type", "unknown"),
                "timestamp": data.get("timestamp", "unknown"),
                "f1_score": data.get("metrics", {}).get("f1_score"),
                "accuracy": data.get("metrics", {}).get("accuracy"),
            })

        return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)
