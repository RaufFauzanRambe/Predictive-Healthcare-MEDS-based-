"""
meds_formatter.py - Convert healthcare data to MEDS (Medical Event Data Standard) format.

The MEDS format provides a standardized, flat representation of longitudinal
medical events. Each row represents a single event for a patient with a code,
timestamp, and optional numeric value. This module converts wide-format
healthcare records into this standardized structure.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class MEDSFormatter:
    """
    Convert wide-format healthcare data into MEDS format.

    MEDS (Medical Event Data Standard) is a flat event-based schema where
    each row represents a single medical event with:
    - subject_id: Patient identifier
    - timestamp: When the event occurred
    - code: Categorical event code (e.g., DIAG/CHF, MED/metformin)
    - numeric_value: Optional numeric measurement

    This formatter handles the mapping from wide columnar data to the
    longitudinal MEDS structure, enabling downstream sequence modeling.

    Attributes:
        config: MEDS configuration dictionary from data_config.yaml.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Initialize the MEDSFormatter.

        Args:
            config: MEDS configuration dict. If None, loads from data_config.yaml.
        """
        if config is None:
            data_config = load_config("configs/data_config.yaml")
            config = data_config.get("meds", {})

        self.config = config

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a wide-format DataFrame into MEDS format.

        The conversion process:
        1. Identify measurement columns (vitals, labs) → code + numeric_value rows
        2. Identify diagnosis columns → code-only rows
        3. Identify medication/procedure columns → code-only rows
        4. Map column names to MEDS codes using configured prefixes
        5. Sort by patient and timestamp
        6. Optionally deduplicate events

        Args:
            df: Wide-format healthcare DataFrame with one row per observation.

        Returns:
            DataFrame in MEDS format with columns: subject_id, timestamp,
            code, numeric_value, plus any configured metadata fields.
        """
        logger.info(f"Converting {len(df)} records to MEDS format")

        code_mappings = self.config.get("code_mappings", {})
        patient_id_field = self.config.get("patient_id_field", "subject_id")
        timestamp_field = self.config.get("timestamp_field", "timestamp")
        code_field = self.config.get("code_field", "code")
        numeric_value_field = self.config.get("numeric_value_field", "numeric_value")

        # Map source column names to patient_id and timestamp
        src_patient_id = "patient_id"
        src_timestamp = "observation_date"

        meds_records: list[dict] = []

        # ── Numeric measurement columns → code + numeric_value rows ──────────
        numeric_measurements = {
            "bmi": f"{code_mappings.get('vital', 'VITAL/')}BMI",
            "blood_pressure_systolic": f"{code_mappings.get('vital', 'VITAL/')}BP_SYSTOLIC",
            "blood_pressure_diastolic": f"{code_mappings.get('vital', 'VITAL/')}BP_DIASTOLIC",
            "heart_rate": f"{code_mappings.get('vital', 'VITAL/')}HEART_RATE",
            "respiratory_rate": f"{code_mappings.get('vital', 'VITAL/')}RESP_RATE",
            "oxygen_saturation": f"{code_mappings.get('vital', 'VITAL/')}O2_SAT",
            "temperature": f"{code_mappings.get('vital', 'VITAL/')}TEMPERATURE",
            "white_blood_cell_count": f"{code_mappings.get('lab', 'LAB/')}WBC",
            "hemoglobin": f"{code_mappings.get('lab', 'LAB/')}HEMOGLOBIN",
            "platelet_count": f"{code_mappings.get('lab', 'LAB/')}PLATELETS",
            "creatinine": f"{code_mappings.get('lab', 'LAB/')}CREATININE",
            "glucose": f"{code_mappings.get('lab', 'LAB/')}GLUCOSE",
        }

        # ── Categorical event columns → code-only rows ───────────────────────
        categorical_events = {
            "admission_type": code_mappings.get("procedure", "PROC/"),
            "insurance_type": code_mappings.get("procedure", "PROC/"),
            "primary_diagnosis": code_mappings.get("diagnosis", "DIAG/"),
            "discharge_disposition": code_mappings.get("procedure", "PROC/"),
        }

        # ── Binary indicator columns → code rows with value 1 ───────────────
        binary_indicators = {
            "has_diabetes": f"{code_mappings.get('diagnosis', 'DIAG/')}DIABETES",
            "has_hypertension": f"{code_mappings.get('diagnosis', 'DIAG/')}HYPERTENSION",
            "has_chd": f"{code_mappings.get('diagnosis', 'DIAG/')}CHD",
            "has_copd": f"{code_mappings.get('diagnosis', 'DIAG/')}COPD",
            "has_renal_disease": f"{code_mappings.get('diagnosis', 'DIAG/')}RENAL",
            "smoker": f"{code_mappings.get('diagnosis', 'DIAG/')}SMOKER",
        }

        # Process each row in the source data
        for _, row in df.iterrows():
            pid = row.get(src_patient_id, "UNKNOWN")
            ts = row.get(src_timestamp, pd.NaT)

            # Numeric measurements
            for col, code in numeric_measurements.items():
                if col in row and pd.notna(row[col]):
                    meds_records.append({
                        patient_id_field: pid,
                        timestamp_field: ts,
                        code_field: code,
                        numeric_value_field: float(row[col]),
                    })

            # Categorical events (value stored in code)
            for col, prefix in categorical_events.items():
                if col in row and pd.notna(row[col]):
                    meds_records.append({
                        patient_id_field: pid,
                        timestamp_field: ts,
                        code_field: f"{prefix}{row[col]}",
                        numeric_value_field: None,
                    })

            # Binary indicators (only record if present/true)
            for col, code in binary_indicators.items():
                if col in row and row[col] == 1:
                    meds_records.append({
                        patient_id_field: pid,
                        timestamp_field: ts,
                        code_field: code,
                        numeric_value_field: 1.0,
                    })

        meds_df = pd.DataFrame(meds_records)

        # Sort by patient and timestamp
        meds_df = meds_df.sort_values(
            [patient_id_field, timestamp_field]
        ).reset_index(drop=True)

        # Deduplicate if configured
        if self.config.get("grouping", {}).get("deduplicate", True):
            before = len(meds_df)
            meds_df = meds_df.drop_duplicates(
                subset=[patient_id_field, timestamp_field, code_field]
            ).reset_index(drop=True)
            after = len(meds_df)
            if before != after:
                logger.info(f"Deduplicated: {before} → {after} records")

        logger.info(f"MEDS conversion complete: {len(meds_df)} events for "
                     f"{meds_df[patient_id_field].nunique()} patients")
        return meds_df

    def save(self, meds_df: pd.DataFrame, output_dir: str = "data/meds_format") -> Path:
        """
        Save MEDS-formatted data to disk as partitioned Parquet files.

        Each patient's data is saved as a separate Parquet file for efficient
        distributed reading. A metadata summary is also saved.

        Args:
            meds_df: MEDS-format DataFrame.
            output_dir: Directory to save the partitioned files.

        Returns:
            Path to the output directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        patient_id_field = self.config.get("patient_id_field", "subject_id")

        # Save full dataset as a single Parquet file
        full_path = output_path / "meds_data.parquet"
        meds_df.to_parquet(full_path, index=False)
        logger.info(f"Saved MEDS data to {full_path}")

        # Save per-patient partitioned files
        partitions_dir = output_path / "patients"
        partitions_dir.mkdir(exist_ok=True)

        for pid, group in meds_df.groupby(patient_id_field):
            safe_pid = str(pid).replace("/", "_")
            patient_path = partitions_dir / f"{safe_pid}.parquet"
            group.to_parquet(patient_path, index=False)

        # Save metadata summary
        metadata = {
            "schema_version": self.config.get("schema_version", "0.3"),
            "total_events": len(meds_df),
            "total_patients": meds_df[patient_id_field].nunique(),
            "unique_codes": meds_df["code"].nunique(),
            "date_range": (
                f"{meds_df[patient_id_field].min()} to "
                f"{meds_df[patient_id_field].max()}"
            ),
        }
        metadata_path = output_path / "metadata.json"
        pd.Series(metadata).to_json(metadata_path, indent=2)

        logger.info(f"Saved partitioned data for {metadata['total_patients']} patients")
        return output_path

    def validate(self, meds_df: pd.DataFrame) -> bool:
        """
        Validate that a DataFrame conforms to the MEDS schema.

        Checks for required columns, non-null patient IDs and timestamps,
        and reasonable data types.

        Args:
            meds_df: MEDS-format DataFrame to validate.

        Returns:
            True if the data passes all validation checks.
        """
        required_columns = [
            self.config.get("patient_id_field", "subject_id"),
            self.config.get("timestamp_field", "timestamp"),
            self.config.get("code_field", "code"),
            self.config.get("numeric_value_field", "numeric_value"),
        ]

        # Check required columns
        missing_cols = [c for c in required_columns if c not in meds_df.columns]
        if missing_cols:
            logger.error(f"MEDS validation failed: missing columns {missing_cols}")
            return False

        # Check for null patient IDs
        pid_col = self.config.get("patient_id_field", "subject_id")
        if meds_df[pid_col].isnull().any():
            logger.error("MEDS validation failed: null patient IDs found")
            return False

        # Check timestamps are datetime
        ts_col = self.config.get("timestamp_field", "timestamp")
        if not pd.api.types.is_datetime64_any_dtype(meds_df[ts_col]):
            logger.warning("Timestamp column is not datetime type; attempting conversion")
            try:
                meds_df[ts_col] = pd.to_datetime(meds_df[ts_col])
            except (ValueError, TypeError):
                logger.error("MEDS validation failed: cannot parse timestamps")
                return False

        logger.info("MEDS validation passed")
        return True
