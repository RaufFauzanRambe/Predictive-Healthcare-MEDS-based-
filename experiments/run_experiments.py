#!/usr/bin/env python
"""
run_experiments.py - CLI entry point untuk menjalankan eksperimen.

Menyediakan antarmuka command-line untuk menjalankan berbagai jenis
eksperimen yang didefinisikan di experiment_configs.yaml, atau
menggunakan parameter custom langsung dari command line.

Usage:
    # Jalankan semua eksperimen dari config
    python experiments/run_experiments.py --all

    # Jalankan eksperimen tertentu
    python experiments/run_experiments.py --experiment lstm_hyperparameter_sweep
    python experiments/run_experiments.py --experiment model_comparison

    # Jalankan single model dengan parameter custom
    python experiments/run_experiments.py --model lstm --lr 0.0005 --epochs 30

    # Smoke test cepat
    python experiments/run_experiments.py --smoke-test

    # List eksperimen yang tersimpan
    python experiments/run_experiments.py --list
"""

import argparse
import sys
from pathlib import Path

# Tambahkan project root ke path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.experiment_runner import ExperimentRunner
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, setup_logging_from_config

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse argumen command-line."""
    parser = argparse.ArgumentParser(
        description="Jalankan eksperimen predictive-healthcare-meds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python experiments/run_experiments.py --experiment model_comparison
  python experiments/run_experiments.py --model lstm --lr 0.0005 --epochs 30
  python experiments/run_experiments.py --smoke-test
  python experiments/run_experiments.py --list
        """,
    )

    # Mode seleksi eksperimen
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Nama eksperimen dari experiment_configs.yaml",
    )
    group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Jalankan semua eksperimen dari config",
    )
    group.add_argument(
        "--model", "-m",
        type=str,
        choices=["mlp", "lstm", "transformer"],
        default=None,
        help="Jalankan single experiment dengan tipe model tertentu",
    )
    group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Jalankan smoke test cepat (verifikasi pipeline)",
    )
    group.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_experiments",
        help="List semua eksperimen yang tersimpan",
    )

    # Override parameter
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Konfigurasi path
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Direktori konfigurasi (default: configs/)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments",
        help="Direktori output eksperimen (default: experiments/)",
    )

    return parser.parse_args()


def build_overrides_from_args(args: argparse.Namespace) -> dict:
    """
    Bangun dictionary override dari argumen CLI.

    Menggabungkan parameter yang diberikan via command line
    ke format nested dict yang kompatibel dengan ExperimentRunner.
    """
    overrides = {}
    training_overrides = {}

    if args.lr is not None:
        training_overrides["learning_rate"] = args.lr
    if args.epochs is not None:
        training_overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        training_overrides["batch_size"] = args.batch_size

    if training_overrides:
        overrides["training"] = training_overrides

    return overrides


def run_named_experiment(
    runner: ExperimentRunner,
    experiment_name: str,
    experiment_configs: dict,
    cli_overrides: dict,
) -> None:
    """
    Jalankan eksperimen berdasarkan nama dari experiment_configs.yaml.

    Mengidentifikasi tipe eksperimen (model_comparison, hyperparameter_sweep,
    ablation_study) dan memanggil method yang sesuai pada ExperimentRunner.
    """
    if experiment_name not in experiment_configs:
        logger.error(f"Eksperimen '{experiment_name}' tidak ditemukan.")
        logger.info(f"Eksperimen yang tersedia: {list(experiment_configs.keys())}")
        sys.exit(1)

    config = experiment_configs[experiment_name]
    exp_type = config.get("experiment_type")

    logger.info(f"Menjalankan eksperimen: {config.get('name', experiment_name)}")
    logger.info(f"Tipe: {exp_type}")
    logger.info(f"Deskripsi: {config.get('description', 'N/A')}")

    if exp_type == "model_comparison":
        model_types = config.get("model_types", ["mlp", "lstm", "transformer"])
        overrides_per_model = config.get("overrides", {})

        # Terapkan CLI overrides ke setiap model
        if cli_overrides:
            for mt in model_types:
                if mt in overrides_per_model:
                    overrides_per_model[mt] = {
                        **overrides_per_model[mt],
                        **cli_overrides,
                    }
                else:
                    overrides_per_model[mt] = cli_overrides

        runner.run_model_comparison(
            model_types=model_types,
            overrides_per_model=overrides_per_model,
        )

    elif exp_type == "hyperparameter_sweep":
        model_type = config.get("model_type", "lstm")
        param_grid = config.get("param_grid", {})
        metric = config.get("metric_to_optimize", "f1_score")

        runner.run_hyperparameter_sweep(
            model_type=model_type,
            param_grid=param_grid,
            metric_to_optimize=metric,
        )

    elif exp_type == "ablation_study":
        model_type = config.get("model_type", "lstm")
        ablation_variants = config.get("ablation_variants", {})

        runner.run_ablation_study(
            model_type=model_type,
            ablation_config=ablation_variants,
        )

    else:
        logger.error(f"Tipe eksperimen tidak dikenal: {exp_type}")
        sys.exit(1)


def main() -> None:
    """Main entry point untuk CLI eksperimen."""
    args = parse_args()

    # Setup
    global_config = load_config(f"{args.config_dir}/config.yaml")
    setup_logging_from_config(global_config)

    logger.info("=" * 70)
    logger.info("PREDICTIVE HEALTHCARE MEDS — Experiment Runner")
    logger.info("=" * 70)

    # Inisialisasi runner
    runner = ExperimentRunner(
        global_config_path=f"{args.config_dir}/config.yaml",
        model_config_path=f"{args.config_dir}/model_config.yaml",
        data_config_path=f"{args.config_dir}/data_config.yaml",
        experiment_dir=args.experiment_dir,
    )

    # Set seed
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Mode: List experiments ────────────────────────────────────────────────
    if args.list_experiments:
        experiments = runner.list_experiments()
        if not experiments:
            print("Belum ada eksperimen yang tersimpan.")
        else:
            print(f"\n{'Run Name':40s} | {'Model':12s} | {'F1':8s} | {'Accuracy':8s} | Timestamp")
            print("-" * 95)
            for exp in experiments:
                f1 = exp.get("f1_score", "N/A")
                acc = exp.get("accuracy", "N/A")
                f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
                acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
                print(f"{exp['run_name']:40s} | {exp['model_type']:12s} | "
                      f"{f1_str:8s} | {acc_str:8s} | {exp['timestamp']}")
        return

    # Bangun CLI overrides
    cli_overrides = build_overrides_from_args(args)

    # ── Mode: Smoke test ──────────────────────────────────────────────────────
    if args.smoke_test:
        logger.info("Menjalankan SMOKE TEST...")
        runner.run_model_comparison(
            model_types=["mlp", "lstm"],
            overrides_per_model={
                "mlp": {"training": {"epochs": 2, "batch_size": 32}},
                "lstm": {"training": {"epochs": 2, "batch_size": 32}},
            },
        )
        logger.info("Smoke test selesai!")
        return

    # ── Mode: Single model ────────────────────────────────────────────────────
    if args.model:
        result = runner.run_single(
            model_type=args.model,
            overrides=cli_overrides if cli_overrides else None,
        )
        logger.info(f"Hasil: F1={result['metrics']['f1_score']:.4f}, "
                     f"Accuracy={result['metrics']['accuracy']:.4f}")
        return

    # ── Mode: Named experiment ────────────────────────────────────────────────
    if args.experiment:
        experiment_configs = load_config("experiments/experiment_configs.yaml")
        run_named_experiment(runner, args.experiment, experiment_configs, cli_overrides)
        return

    # ── Mode: All experiments ─────────────────────────────────────────────────
    if args.all:
        experiment_configs = load_config("experiments/experiment_configs.yaml")
        logger.info(f"Menjalankan SEMUA eksperimen: {list(experiment_configs.keys())}")

        for name, config in experiment_configs.items():
            logger.info(f"\n{'#' * 70}")
            logger.info(f"EKSPERIMEN: {name}")
            logger.info(f"{'#' * 70}")
            try:
                run_named_experiment(runner, name, experiment_configs, cli_overrides)
            except Exception as e:
                logger.error(f"Eksperimen '{name}' gagal: {e}")
                continue

        logger.info("Semua eksperimen selesai!")
        return

    # Tidak ada mode yang dipilih — tampilkan help
    logger.info("Gunakan --help untuk melihat opsi yang tersedia.")
    logger.info("Contoh: python experiments/run_experiments.py --experiment model_comparison")


if __name__ == "__main__":
    main()
