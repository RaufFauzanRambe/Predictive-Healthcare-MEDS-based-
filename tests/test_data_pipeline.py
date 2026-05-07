"""
test_data_pipeline.py - Unit tests for the data loading and preprocessing pipeline.

Tests cover data generation, missing value imputation, outlier handling,
normalization, and MEDS format conversion.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.preprocess import Preprocessor
from src.data_pipeline.meds_formatter import MEDSFormatter


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def data_config():
    """Provide a minimal data configuration for testing."""
    return {
        "raw_data": {"dir": "data/raw"},
        "dummy_data": {
            "enabled": True,
            "num_patients": 50,
            "num_events_per_patient": 10,
            "positive_rate": 0.25,
            "time_span_days": 90,
        },
        "columns": {
            "numeric": [
                "age", "bmi", "blood_pressure_systolic", "heart_rate",
                "oxygen_saturation", "creatinine", "glucose",
            ],
            "categorical": ["gender", "admission_type", "primary_diagnosis"],
            "binary": ["has_diabetes", "has_hypertension", "smoker"],
        },
        "preprocessing": {
            "missing_values": {
                "strategy": "median",
                "max_missing_ratio": 0.8,
                "indicator_features": True,
            },
            "outliers": {
                "method": "iqr",
                "iqr_factor": 1.5,
                "action": "clip",
            },
            "normalization": {
                "method": "standard",
            },
            "encoding": {
                "categorical_method": "label",
                "max_categories": 20,
            },
            "time_series": {
                "resample_freq": "1H",
                "interpolation": "linear",
            },
        },
        "meds": {
            "schema_version": "0.3",
            "patient_id_field": "subject_id",
            "timestamp_field": "timestamp",
            "code_field": "code",
            "numeric_value_field": "numeric_value",
            "code_mappings": {
                "diagnosis": "DIAG/",
                "medication": "MED/",
                "procedure": "PROC/",
                "lab": "LAB/",
                "vital": "VITAL/",
            },
            "grouping": {"deduplicate": True},
        },
    }


@pytest.fixture
def sample_dataframe():
    """Create a small sample DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in rng.integers(0, 20, n)],
        "event_id": [f"E{i:06d}" for i in range(n)],
        "observation_date": pd.date_range("2023-01-01", periods=n, freq="6h"),
        "admission_date": pd.date_range("2023-01-01", periods=n, freq="6h"),
        "discharge_date": pd.date_range("2023-01-05", periods=n, freq="6h"),
        "age": rng.integers(18, 90, n),
        "gender": rng.choice(["M", "F"], n),
        "bmi": rng.normal(28, 5, n),
        "blood_pressure_systolic": rng.normal(130, 20, n),
        "heart_rate": rng.normal(78, 12, n),
        "oxygen_saturation": rng.normal(96, 3, n),
        "creatinine": rng.normal(1.1, 0.3, n),
        "glucose": rng.normal(120, 40, n),
        "admission_type": rng.choice(["Emergency", "Urgent", "Elective"], n),
        "primary_diagnosis": rng.choice(["CHF", "Pneumonia", "COPD"], n),
        "has_diabetes": rng.integers(0, 2, n),
        "has_hypertension": rng.integers(0, 2, n),
        "has_chd": rng.integers(0, 2, n),
        "has_copd": rng.integers(0, 2, n),
        "has_renal_disease": rng.integers(0, 2, n),
        "smoker": rng.integers(0, 2, n),
        "length_of_stay": rng.integers(1, 14, n),
        "num_prior_admissions": rng.integers(0, 5, n),
        "num_medications": rng.integers(1, 15, n),
        "num_procedures": rng.integers(0, 5, n),
        "num_diagnoses": rng.integers(1, 8, n),
        "insurance_type": rng.choice(["Medicare", "Private", "Medicaid"], n),
        "discharge_disposition": rng.choice(["Home", "SNF", "Rehab"], n),
        "icu_stay": rng.choice(["Y", "N"], n),
        "readmission_30day": rng.integers(0, 2, n),
    })


# ─── DataLoader Tests ────────────────────────────────────────────────────────

class TestDataLoader:
    """Tests for the DataLoader class."""

    def test_generate_dummy_data(self, data_config):
        """Test that dummy data generation produces valid output."""
        loader = DataLoader(data_config)
        df = loader.generate_dummy_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "patient_id" in df.columns
        assert "readmission_30day" in df.columns
        assert df["readmission_30day"].isin([0, 1]).all()

    def test_dummy_data_has_missing_values(self, data_config):
        """Test that dummy data introduces realistic missingness."""
        loader = DataLoader(data_config)
        df = loader.generate_dummy_data()

        # Should have some NaN values in selected columns
        missing_cols = df.isnull().sum()
        assert missing_cols.sum() > 0, "Dummy data should contain some missing values"

    def test_dummy_data_patient_count(self, data_config):
        """Test that generated data has the expected number of patients."""
        expected_patients = data_config["dummy_data"]["num_patients"]
        loader = DataLoader(data_config)
        df = loader.generate_dummy_data()

        assert df["patient_id"].nunique() == expected_patients

    def test_save_and_load_csv(self, data_config, tmp_path):
        """Test saving and reloading data in CSV format."""
        loader = DataLoader(data_config)
        df = loader.generate_dummy_data()

        filepath = tmp_path / "test_data.csv"
        loader.save(df, filepath, format="csv")

        assert filepath.exists()
        loaded_df = loader._read_file(filepath)
        assert len(loaded_df) == len(df)


# ─── Preprocessor Tests ─────────────────────────────────────────────────────

class TestPreprocessor:
    """Tests for the Preprocessor class."""

    def test_fit_transform_completes(self, data_config, sample_dataframe):
        """Test that fit_transform runs without error."""
        preprocessor = Preprocessor(data_config["preprocessing"])
        columns_config = data_config["columns"]

        result = preprocessor.fit_transform(sample_dataframe, columns_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)

    def test_missing_values_imputed(self, data_config):
        """Test that missing values are properly imputed."""
        df = pd.DataFrame({
            "age": [25, np.nan, 45, 60, np.nan],
            "bmi": [22.0, 28.0, np.nan, 30.0, 25.0],
            "blood_pressure_systolic": [120, 140, 130, np.nan, 110],
            "heart_rate": [72, 80, 76, 88, 84],
            "oxygen_saturation": [98, 96, 97, 95, 99],
            "creatinine": [1.0, 1.2, np.nan, 0.9, 1.1],
            "glucose": [100, 140, 120, np.nan, 110],
            "gender": ["M", "F", "M", "F", "M"],
            "admission_type": ["Emergency", "Elective", "Urgent", "Emergency", "Elective"],
            "primary_diagnosis": ["CHF", "COPD", "Pneumonia", "CHF", "COPD"],
            "has_diabetes": [0, 1, 0, 1, 0],
            "has_hypertension": [1, 0, 1, 0, 1],
            "smoker": [0, 0, 1, 0, 1],
        })

        preprocessor = Preprocessor(data_config["preprocessing"])
        columns_config = data_config["columns"]
        result = preprocessor.fit_transform(df, columns_config)

        # Numeric columns should have no missing values after imputation
        numeric_cols = [c for c in columns_config["numeric"] if c in result.columns]
        assert result[numeric_cols].isnull().sum().sum() == 0

    def test_outliers_clipped(self, data_config):
        """Test that outliers are clipped when using IQR method."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 200],  # 200 is an extreme outlier
            "bmi": [22.0, 24.0, 26.0, 28.0, 60.0],  # 60 is extreme
            "blood_pressure_systolic": [120, 125, 130, 135, 300],
            "heart_rate": [72, 75, 78, 80, 200],
            "oxygen_saturation": [97, 96, 95, 94, 50],
            "creatinine": [0.8, 1.0, 1.2, 1.4, 10.0],
            "glucose": [90, 110, 130, 150, 500],
            "gender": ["M", "F", "M", "F", "M"],
            "admission_type": ["Emergency"] * 5,
            "primary_diagnosis": ["CHF"] * 5,
            "has_diabetes": [0, 0, 0, 0, 0],
            "has_hypertension": [0, 0, 0, 0, 0],
            "smoker": [0, 0, 0, 0, 0],
        })

        preprocessor = Preprocessor(data_config["preprocessing"])
        columns_config = data_config["columns"]
        result = preprocessor.fit_transform(df, columns_config)

        # After clipping, extreme values should be capped
        # The exact cap depends on IQR, but 200 for age should definitely be clipped
        assert result["age"].max() < 200

    def test_transform_requires_fit(self, data_config, sample_dataframe):
        """Test that transform() raises error before fit_transform()."""
        preprocessor = Preprocessor(data_config["preprocessing"])

        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(sample_dataframe)


# ─── MEDS Formatter Tests ────────────────────────────────────────────────────

class TestMEDSFormatter:
    """Tests for the MEDSFormatter class."""

    def test_convert_produces_meds_format(self, data_config, sample_dataframe):
        """Test that conversion produces the correct MEDS columns."""
        formatter = MEDSFormatter(data_config["meds"])
        result = formatter.convert(sample_dataframe)

        assert "subject_id" in result.columns
        assert "timestamp" in result.columns
        assert "code" in result.columns
        assert "numeric_value" in result.columns

    def test_convert_has_more_rows_than_input(self, data_config, sample_dataframe):
        """Test that MEDS format has more rows (one per event type) than input."""
        formatter = MEDSFormatter(data_config["meds"])
        result = formatter.convert(sample_dataframe)

        # Each input row produces multiple MEDS events (vitals + labs + diagnoses)
        assert len(result) > len(sample_dataframe)

    def test_validate_passes_for_valid_data(self, data_config, sample_dataframe):
        """Test that validation passes for properly converted data."""
        formatter = MEDSFormatter(data_config["meds"])
        meds_df = formatter.convert(sample_dataframe)
        assert formatter.validate(meds_df) is True

    def test_code_prefixes_applied(self, data_config, sample_dataframe):
        """Test that MEDS codes have the correct prefixes."""
        formatter = MEDSFormatter(data_config["meds"])
        result = formatter.convert(sample_dataframe)

        # Should have codes starting with DIAG/, VITAL/, LAB/
        codes = result["code"].unique()
        has_diag = any(c.startswith("DIAG/") for c in codes)
        has_vital = any(c.startswith("VITAL/") for c in codes)
        has_lab = any(c.startswith("LAB/") for c in codes)

        assert has_diag, "Should have diagnosis codes"
        assert has_vital, "Should have vital sign codes"
        assert has_lab, "Should have lab codes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
