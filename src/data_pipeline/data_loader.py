"""
data_loader.py - Load raw healthcare data from CSV, JSON, or Parquet files.

This module provides a unified DataLoader class that handles reading healthcare
data from multiple formats, validating schema, and generating synthetic data
when real data is not available.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class DataLoader:
    """
    Unified data loader for healthcare datasets.

    Supports CSV, JSON, and Parquet file formats. When no data files are found,
    it can generate synthetic/dummy healthcare data for development and testing
    purposes based on the data_config.yaml settings.

    Attributes:
        data_config: Configuration dictionary for data settings.
        raw_data_dir: Path to the raw data directory.
    """

    def __init__(self, data_config: Optional[dict] = None) -> None:
        """
        Initialize the DataLoader with optional configuration override.

        Args:
            data_config: Optional data configuration dict. If None, loads from
                         the default data_config.yaml.
        """
        if data_config is None:
            data_config = load_config("configs/data_config.yaml")
        self.data_config = data_config
        self.raw_data_dir = Path(data_config.get("raw_data", {}).get("dir", "data/raw"))

    def load(self, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load healthcare data from a file or generate dummy data.

        If a filepath is provided, the loader will detect the file format from
        the extension and read accordingly. If no filepath is given, it searches
        the raw data directory for any compatible files. If no files are found,
        synthetic data is generated automatically.

        Args:
            filepath: Optional path to a specific data file.

        Returns:
            A pandas DataFrame containing the loaded or generated data.

        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            ValueError: If the file format is not supported.
        """
        if filepath is not None:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            return self._read_file(filepath)

        # Auto-discover files in raw data directory
        if self.raw_data_dir.exists():
            supported_extensions = {".csv", ".json", ".parquet"}
            files = [
                f
                for f in self.raw_data_dir.iterdir()
                if f.suffix.lower() in supported_extensions
            ]
            if files:
                logger.info(f"Found {len(files)} data file(s) in {self.raw_data_dir}")
                return self._read_file(files[0])

        # No data found — generate dummy data
        logger.warning("No data files found. Generating synthetic healthcare data...")
        return self.generate_dummy_data()

    def _read_file(self, filepath: Path) -> pd.DataFrame:
        """
        Read a data file based on its extension.

        Supports CSV, JSON, and Parquet formats. JSON files can be either
        line-delimited (.jsonl) or standard JSON arrays.

        Args:
            filepath: Path to the data file.

        Returns:
            A pandas DataFrame with the file contents.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = filepath.suffix.lower()
        logger.info(f"Loading data from {filepath} (format: {ext})")

        if ext == ".csv":
            df = pd.read_csv(filepath, parse_dates=True)
        elif ext == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
            # Handle both list-of-records and nested structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def generate_dummy_data(self) -> pd.DataFrame:
        """
        Generate synthetic healthcare data for development and testing.

        Creates a realistic-looking dataset with patient demographics, vital
        signs, lab values, and a binary readmission target. The data follows
        configurable distributions to simulate real healthcare patterns.

        Returns:
            A pandas DataFrame with synthetic healthcare records.
        """
        cfg = self.data_config.get("dummy_data", {})
        num_patients = cfg.get("num_patients", 1000)
        events_per_patient = cfg.get("num_events_per_patient", 50)
        positive_rate = cfg.get("positive_rate", 0.25)
        time_span_days = cfg.get("time_span_days", 365)

        rng = np.random.default_rng(42)
        records: list[dict[str, Any]] = []

        for pid in range(num_patients):
            base_date = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=rng.integers(0, time_span_days)
            )
            gender = rng.choice(["M", "F"])
            race = rng.choice(["White", "Black", "Hispanic", "Asian", "Other"],
                               p=[0.5, 0.2, 0.15, 0.1, 0.05])
            age = int(rng.normal(62, 18))
            age = max(1, min(age, 99))

            for event_idx in range(rng.integers(1, events_per_patient + 1)):
                obs_date = base_date + pd.Timedelta(
                    hours=rng.integers(0, time_span_days * 24)
                )
                record = {
                    "patient_id": f"P{pid:06d}",
                    "event_id": f"E{pid:06d}_{event_idx:04d}",
                    "observation_date": obs_date,
                    "admission_date": obs_date,
                    "discharge_date": obs_date + pd.Timedelta(days=rng.integers(1, 14)),
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "bmi": round(float(rng.normal(28, 6)), 1),
                    "blood_pressure_systolic": int(rng.normal(130, 20)),
                    "blood_pressure_diastolic": int(rng.normal(80, 12)),
                    "heart_rate": int(rng.normal(78, 15)),
                    "respiratory_rate": int(rng.normal(18, 4)),
                    "oxygen_saturation": round(float(rng.normal(96, 3)), 1),
                    "temperature": round(float(rng.normal(98.6, 1.0)), 1),
                    "white_blood_cell_count": round(float(rng.normal(8, 3)), 1),
                    "hemoglobin": round(float(rng.normal(13, 2)), 1),
                    "platelet_count": int(rng.normal(250, 80)),
                    "creatinine": round(float(rng.normal(1.1, 0.4)), 2),
                    "glucose": int(rng.normal(120, 40)),
                    "length_of_stay": int(rng.exponential(4)),
                    "num_prior_admissions": int(rng.poisson(2)),
                    "num_medications": int(rng.poisson(6)),
                    "num_procedures": int(rng.poisson(1.5)),
                    "num_diagnoses": int(rng.poisson(4)),
                    "admission_type": rng.choice(["Emergency", "Urgent", "Elective"],
                                                  p=[0.5, 0.3, 0.2]),
                    "insurance_type": rng.choice(["Medicare", "Medicaid", "Private", "SelfPay"],
                                                  p=[0.45, 0.2, 0.3, 0.05]),
                    "primary_diagnosis": rng.choice([
                        "CHF", "Pneumonia", "COPD", "AMI", "Stroke",
                        "Sepsis", "Fracture", "DKA", "PE", "GI_Bleed",
                    ]),
                    "discharge_disposition": rng.choice(["Home", "SNF", "Rehab", "AMA"],
                                                         p=[0.6, 0.2, 0.15, 0.05]),
                    "icu_stay": rng.choice(["Y", "N"], p=[0.3, 0.7]),
                    "has_diabetes": int(rng.random() < 0.3),
                    "has_hypertension": int(rng.random() < 0.45),
                    "has_chd": int(rng.random() < 0.2),
                    "has_copd": int(rng.random() < 0.15),
                    "has_renal_disease": int(rng.random() < 0.1),
                    "smoker": int(rng.random() < 0.25),
                    "readmission_30day": int(rng.random() < positive_rate),
                }
                records.append(record)

        df = pd.DataFrame(records)

        # Introduce realistic missing values (~5% random missingness)
        for col in ["bmi", "blood_pressure_systolic", "oxygen_saturation",
                     "creatinine", "glucose", "hemoglobin"]:
            mask = rng.random(len(df)) < 0.05
            df.loc[mask, col] = np.nan

        logger.info(f"Generated {len(df)} synthetic records for {num_patients} patients")
        return df

    def save(self, df: pd.DataFrame, filepath: Union[str, Path], format: str = "csv") -> None:
        """
        Save a DataFrame to disk.

        Args:
            df: The DataFrame to save.
            filepath: Destination file path.
            format: Output format — 'csv', 'json', or 'parquet'.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Saved {len(df)} records to {filepath}")
