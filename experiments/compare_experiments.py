#!/usr/bin/env python
"""
compare_experiments.py - Perbandingan dan analisis hasil eksperimen.

Menyediakan tools untuk:
- Memuat dan membandingkan hasil eksperimen yang tersimpan
- Menghasilkan tabel perbandingan di console
- Menganalisis ranking dan tren hyperparameter sweep
- Menghasilkan laporan perbandingan dalam format JSON

Usage:
    # Bandingkan dua eksperimen
    python experiments/compare_experiments.py --runs comparison_mlp comparison_lstm

    # Tampilkan semua hasil sweep
    python experiments/compare_experiments.py --sweep sweep_lstm_results

    # Tampilkan top-N konfigurasi terbaik
    python experiments/compare_experiments.py --top 5

    # Laporan lengkap semua eksperimen
    python experiments/compare_experiments.py --report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Tambahkan project root ke path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.experiment_runner import ExperimentRunner
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =========================================================================
# Formatting & Display Helpers
# =========================================================================

class ExperimentComparator:
    """
    Komparator untuk menganalisis dan membandingkan hasil eksperimen.

    Memuat file JSON hasil eksperimen, mengekstrak metrik, dan
    menyajikan perbandingan dalam format tabel yang mudah dibaca.
    """

    def __init__(self, experiment_dir: str = "experiments") -> None:
        self.experiment_dir = Path(experiment_dir)
        self.results_dir = self.experiment_dir / "results"

    def load_run(self, run_name: str) -> Optional[dict]:
        """Muat hasil eksperimen dari file JSON."""
        filepath = self.results_dir / f"{run_name}.json"
        if not filepath.exists():
            # Coba cari di root experiment dir
            filepath = self.experiment_dir / f"{run_name}.json"
        if not filepath.exists():
            logger.error(f"File hasil tidak ditemukan: {run_name}")
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def load_sweep_results(self, sweep_name: str) -> Optional[dict]:
        """Muat hasil hyperparameter sweep."""
        filepath = self.experiment_dir / f"{sweep_name}.json"
        if not filepath.exists():
            logger.error(f"File sweep tidak ditemukan: {sweep_name}")
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def compare_runs(self, run_names: list[str]) -> None:
        """
        Tampilkan perbandingan metrik dari beberapa run.

        Memuat setiap file JSON hasil eksperimen dan mencetak
        tabel perbandingan yang mencakup semua metrik utama.
        """
        all_data = []
        for name in run_names:
            data = self.load_run(name)
            if data:
                all_data.append(data)

        if not all_data:
            print("Tidak ada data eksperimen yang ditemukan.")
            return

        # Header
        print("\n" + "=" * 120)
        print("PERBANDINGAN HASIL EKSPERIMEN")
        print("=" * 120)

        # Column headers
        print(
            f"{'Run Name':35s} | {'Model':12s} | {'Accuracy':9s} | "
            f"{'Precision':9s} | {'Recall':9s} | {'F1':9s} | "
            f"{'AUC-ROC':9s} | {'Params':>10s} | {'Time':8s}"
        )
        print("-" * 120)

        for data in all_data:
            metrics = data.get("metrics", {})
            print(
                f"{data.get('run_name', 'N/A'):35s} | "
                f"{data.get('model_type', 'N/A'):12s} | "
                f"{metrics.get('accuracy', 0):9.4f} | "
                f"{metrics.get('precision', 0):9.4f} | "
                f"{metrics.get('recall', 0):9.4f} | "
                f"{metrics.get('f1_score', 0):9.4f} | "
                f"{metrics.get('auc_roc', 0) if metrics.get('auc_roc') else 'N/A':>9s} | "
                f"{data.get('parameter_count', 0):>10,} | "
                f"{data.get('training_time_seconds', 0):7.1f}s"
            )

        print("=" * 120)

        # Tentukan model terbaik
        best = max(all_data, key=lambda x: x.get("metrics", {}).get("f1_score", 0))
        print(f"\n  MODEL TERBAIK (by F1): {best.get('run_name', 'N/A')}")
        print(f"    F1 Score:  {best['metrics']['f1_score']:.4f}")
        print(f"    Accuracy:  {best['metrics']['accuracy']:.4f}")
        print(f"    AUC-ROC:   {best['metrics'].get('auc_roc', 'N/A')}")
        print()

    def show_sweep_results(self, sweep_name: str, top_n: int = 10) -> None:
        """
        Tampilkan hasil hyperparameter sweep, diurutkan berdasarkan metrik.

        Menampilkan ranking konfigurasi terbaik beserta parameter
        yang digunakan pada setiap run dalam sweep.
        """
        data = self.load_sweep_results(sweep_name)
        if not data:
            return

        results = data.get("all_results", [])
        successful = [r for r in results if "error" not in r]
        metric = data.get("metric_to_optimize", "f1_score")

        # Sort berdasarkan metrik
        successful.sort(key=lambda x: x.get("score", 0), reverse=True)

        print("\n" + "=" * 100)
        print(f"HYPERPARAMETER SWEEP RESULTS: {sweep_name}")
        print(f"Model: {data.get('model_type', 'N/A')}")
        print(f"Total runs: {data.get('total_runs', 0)} | "
              f"Successful: {data.get('successful_runs', 0)}")
        print(f"Best {metric}: {data.get('best_score', 'N/A')}")
        print("=" * 100)

        # Tabel hasil (top-N)
        display_count = min(top_n, len(successful))
        print(f"\nTop {display_count} konfigurasi:\n")

        for i, run in enumerate(successful[:display_count], 1):
            marker = " >>> BEST" if i == 1 else ""
            params_str = " | ".join(f"{k}={v}" for k, v in run.get("params", {}).items())
            print(
                f"  #{i:2d} | {metric}={run.get('score', 0):.4f} | "
                f"Acc={run.get('metrics', {}).get('accuracy', 0):.4f} | "
                f"Time={run.get('training_time', 0):.1f}s | "
                f"{params_str}{marker}"
            )

        # Best config detail
        if data.get("best_config"):
            print(f"\nKonfigurasi terbaik:")
            for k, v in data["best_config"].items():
                print(f"  {k}: {v}")

        print()

    def show_top_runs(self, top_n: int = 5) -> None:
        """Tampilkan top-N eksperimen berdasarkan F1 score."""
        runner = ExperimentRunner(experiment_dir=str(self.experiment_dir))
        all_experiments = runner.list_experiments()

        if not all_experiments:
            print("Belum ada eksperimen yang tersimpan.")
            return

        # Sort by F1 score
        all_experiments.sort(
            key=lambda x: x.get("f1_score", 0) if x.get("f1_score") else 0,
            reverse=True,
        )

        display_count = min(top_n, len(all_experiments))

        print("\n" + "=" * 80)
        print(f"TOP {display_count} EKSPERIMEN (by F1 Score)")
        print("=" * 80)

        for i, exp in enumerate(all_experiments[:display_count], 1):
            f1 = exp.get("f1_score", 0)
            acc = exp.get("accuracy", 0)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else "N/A"
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else "N/A"

            print(f"  #{i} | {exp['run_name']:40s} | "
                  f"F1={f1_str:8s} | Acc={acc_str:8s} | "
                  f"{exp['model_type']:12s} | {exp['timestamp']}")

        print("=" * 80)
        print()

    def generate_report(self) -> dict[str, Any]:
        """
        Hasilkan laporan lengkap semua eksperimen yang tersimpan.

        Mengumpulkan semua file JSON di direktori eksperimen,
        membuat ringkasan statistik, dan menyimpan laporan.
        """
        all_results = []

        # Scan semua file hasil
        if self.results_dir.exists():
            for filepath in self.results_dir.glob("*.json"):
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    all_results.append(data)
                except (json.JSONDecodeError, IOError):
                    continue

        # Scan file sweep dan comparison
        for filepath in self.experiment_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                if data.get("experiment_type") in ("model_comparison", "hyperparameter_sweep", "ablation_study"):
                    all_results.append(data)
            except (json.JSONDecodeError, IOError):
                continue

        if not all_results:
            print("Tidak ada data eksperimen untuk dilaporkan.")
            return {}

        # Statistik ringkasan
        single_runs = [r for r in all_results if "model_type" in r and "metrics" in r]

        report = {
            "total_experiments": len(all_results),
            "total_single_runs": len(single_runs),
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "models_tested": list(set(r.get("model_type", "unknown") for r in single_runs)),
        }

        if single_runs:
            f1_scores = [r["metrics"]["f1_score"] for r in single_runs if "metrics" in r]
            if f1_scores:
                report["best_f1"] = max(f1_scores)
                report["worst_f1"] = min(f1_scores)
                report["mean_f1"] = sum(f1_scores) / len(f1_scores)

            acc_scores = [r["metrics"]["accuracy"] for r in single_runs if "metrics" in r]
            if acc_scores:
                report["best_accuracy"] = max(acc_scores)

        # Simpan laporan
        report_path = self.experiment_dir / "experiment_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Tampilkan ringkasan
        print("\n" + "=" * 60)
        print("LAPORAN EKSPERIMEN")
        print("=" * 60)
        print(f"  Total eksperimen: {report.get('total_experiments', 0)}")
        print(f"  Model yang diuji: {report.get('models_tested', [])}")
        print(f"  Best F1:          {report.get('best_f1', 'N/A')}")
        print(f"  Best Accuracy:    {report.get('best_accuracy', 'N/A')}")
        print(f"  Mean F1:          {report.get('mean_f1', 'N/A'):.4f}" if isinstance(report.get('mean_f1'), float) else f"  Mean F1: N/A")
        print(f"\n  Laporan disimpan: {report_path}")
        print("=" * 60 + "\n")

        return report


def parse_args() -> argparse.Namespace:
    """Parse argumen command-line."""
    parser = argparse.ArgumentParser(
        description="Bandingkan dan analisis hasil eksperimen",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--runs", "-r",
        nargs="+",
        help="Bandingkan hasil dari run names tertentu",
    )
    group.add_argument(
        "--sweep", "-s",
        type=str,
        help="Tampilkan hasil hyperparameter sweep",
    )
    group.add_argument(
        "--top", "-t",
        type=int,
        default=None,
        help="Tampilkan top-N eksperimen terbaik",
    )
    group.add_argument(
        "--report",
        action="store_true",
        help="Hasilkan laporan lengkap semua eksperimen",
    )

    parser.add_argument("--experiment-dir", type=str, default="experiments")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N untuk sweep results")

    return parser.parse_args()


def main() -> None:
    """Entry point utama."""
    args = parse_args()
    comparator = ExperimentComparator(experiment_dir=args.experiment_dir)

    if args.runs:
        comparator.compare_runs(args.runs)
    elif args.sweep:
        comparator.show_sweep_results(args.sweep, top_n=args.top_n)
    elif args.top is not None:
        comparator.show_top_runs(top_n=args.top)
    elif args.report:
        comparator.generate_report()
    else:
        # Default: tampilkan top 10
        comparator.show_top_runs(top_n=10)


if __name__ == "__main__":
    main()
