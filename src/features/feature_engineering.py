"""
feature_engineering.py - Generate derived features for healthcare predictive models.

Creates clinical risk scores, demographic groupings, interaction features,
rolling time-series statistics, and temporal trend features from raw
healthcare data.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for healthcare predictive modeling.

    Generates clinically meaningful derived features including:
    - Age group categorization
    - Comorbidity risk scores (Charlson, Elixhauser)
    - Clinical interaction features
    - Rolling statistics for time-series data
    - Temporal trend (difference) features

    Attributes:
        config: Feature engineering configuration dictionary.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Initialize the FeatureEngineer.

        Args:
            config: Feature engineering config dict. If None, loads from
                    data_config.yaml under the 'feature_engineering' key.
        """
        if config is None:
            data_config = load_config("configs/data_config.yaml")
            config = data_config.get("feature_engineering", {})

        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full feature engineering pipeline.

        Sequentially applies all enabled feature transformations to create
        a rich feature set for downstream modeling.

        Args:
            df: Input DataFrame (should already be preprocessed).

        Returns:
            DataFrame with all engineered features appended.
        """
        logger.info(f"Feature engineering on {df.shape} data")
        df = df.copy()

        if self.config.get("age_groups", True):
            df = self._create_age_groups(df)

        if self.config.get("risk_scores", True):
            df = self._compute_risk_scores(df)

        if self.config.get("interaction_features", True):
            df = self._create_interaction_features(df)

        if self.config.get("rolling_statistics", True):
            df = self._create_rolling_features(df)

        if self.config.get("diff_features", True):
            df = self._create_diff_features(df)

        if self.config.get("polynomial_features", False):
            df = self._create_polynomial_features(df)

        logger.info(f"Feature engineering complete: {df.shape}")
        return df

    def _create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group categories from continuous age values.

        Bins ages into clinically relevant groups (pediatric, young adult,
        adult, middle-aged, senior, elderly) based on configurable bins.
        """
        if "age" not in df.columns:
            logger.warning("'age' column not found; skipping age groups")
            return df

        bins = self.config.get("age_bins", [0, 18, 35, 50, 65, 80, 120])
        labels = self.config.get("age_labels",
                                 ["pediatric", "young_adult", "adult",
                                  "middle_aged", "senior", "elderly"])

        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        # One-hot encode the age group
        age_dummies = pd.get_dummies(df["age_group"], prefix="age_grp", dtype=int)
        df = pd.concat([df, age_dummies], axis=1)
        df = df.drop(columns=["age_group"])

        logger.debug(f"Created age group features: {list(age_dummies.columns)}")
        return df

    def _compute_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute clinical comorbidity risk scores.

        Implements simplified versions of:
        - Charlson Comorbidity Index (CCI): Predicts 10-year mortality
        - Elixhauser Comorbidity Measure: Predicts hospital outcomes

        These scores sum weighted comorbidity indicators to produce a single
        risk score per patient, which is a strong predictor of readmission.
        """
        # Charlson Comorbidity Index (simplified)
        charlson_score = pd.Series(0, index=df.index)

        if "has_diabetes" in df.columns:
            charlson_score += df["has_diabetes"] * 1
        if "has_chd" in df.columns:
            charlson_score += df["has_chd"] * 2
        if "has_copd" in df.columns:
            charlson_score += df["has_copd"] * 1
        if "has_renal_disease" in df.columns:
            charlson_score += df["has_renal_disease"] * 2
        if "has_hypertension" in df.columns:
            charlson_score += df["has_hypertension"] * 1

        # Age contribution to Charlson (1 point per decade over 40)
        if "age" in df.columns:
            age_contribution = ((df["age"] - 40) / 10).clip(lower=0).astype(int)
            charlson_score += age_contribution

        df["charlson_index"] = charlson_score

        # Elixhauser Comorbidity Score (simplified count-based)
        elixhauser_score = pd.Series(0, index=df.index)

        if "has_diabetes" in df.columns:
            elixhauser_score += df["has_diabetes"]
        if "has_hypertension" in df.columns:
            elixhauser_score += df["has_hypertension"]
        if "has_chd" in df.columns:
            elixhauser_score += df["has_chd"]
        if "has_copd" in df.columns:
            elixhauser_score += df["has_copd"]
        if "has_renal_disease" in df.columns:
            elixhauser_score += df["has_renal_disease"]

        df["elixhauser_score"] = elixhauser_score

        logger.debug(f"Risk scores — CCI: mean={df['charlson_index'].mean():.2f}, "
                      f"Elixhauser: mean={df['elixhauser_score'].mean():.2f}")
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinically meaningful interaction features.

        These interactions capture known clinical relationships:
        - Age x Comorbidities: older patients with comorbidities face higher risk
        - BMI x Diabetes: obesity amplifies diabetes complications
        - Heart rate x Blood pressure: hemodynamic instability indicators
        - Length of stay x Prior admissions: healthcare utilization burden
        """
        # Age × comorbidity burden
        if "age" in df.columns and "charlson_index" in df.columns:
            df["age_charlson_interaction"] = df["age"] * df["charlson_index"]

        # BMI × diabetes interaction
        if "bmi" in df.columns and "has_diabetes" in df.columns:
            df["bmi_diabetes_interaction"] = df["bmi"] * df["has_diabetes"]

        # Hemodynamic stress indicator
        if "heart_rate" in df.columns and "blood_pressure_systolic" in df.columns:
            # Rate-pressure product (clinical indicator of cardiac workload)
            df["rate_pressure_product"] = df["heart_rate"] * df["blood_pressure_systolic"] / 1000

        # Healthcare utilization burden
        if "length_of_stay" in df.columns and "num_prior_admissions" in df.columns:
            df["utilization_burden"] = df["length_of_stay"] * (1 + df["num_prior_admissions"])

        # Lab-based severity marker
        if "creatinine" in df.columns and "glucose" in df.columns:
            df["metabolic_stress_index"] = df["creatinine"] * df["glucose"] / 100

        logger.debug("Created interaction features")
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics for time-series features.

        Computes rolling mean, standard deviation, min, and max over
        configurable time windows for numeric columns. Requires the
        data to be sorted by patient and timestamp.
        """
        windows = self.config.get("rolling_windows", [6, 12, 24, 48])
        patient_id_col = "patient_id"

        if patient_id_col not in df.columns:
            logger.warning("Cannot create rolling features: no patient_id column")
            return df

        vital_cols = [c for c in ["heart_rate", "blood_pressure_systolic",
                                    "oxygen_saturation", "respiratory_rate",
                                    "temperature"] if c in df.columns]

        if not vital_cols:
            return df

        for window in windows:
            for col in vital_cols:
                rolling = df.groupby(patient_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_rolling_mean_{window}"] = rolling

                rolling_std = df.groupby(patient_id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[f"{col}_rolling_std_{window}"] = rolling_std.fillna(0)

        logger.debug(f"Created rolling features for {len(vital_cols)} vitals "
                      f"across {len(windows)} windows")
        return df

    def _create_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create first-difference (trend) features for time-series data.

        For each patient, computes the change between consecutive observations
        for vital signs. Large changes can indicate clinical deterioration.
        """
        patient_id_col = "patient_id"

        if patient_id_col not in df.columns:
            return df

        vital_cols = [c for c in ["heart_rate", "blood_pressure_systolic",
                                    "oxygen_saturation", "creatinine",
                                    "glucose", "temperature"] if c in df.columns]

        for col in vital_cols:
            df[f"{col}_diff"] = df.groupby(patient_id_col)[col].diff().fillna(0)
            # Absolute change as a separate feature
            df[f"{col}_abs_diff"] = df[f"{col}_diff"].abs()

        logger.debug(f"Created diff features for {len(vital_cols)} columns")
        return df

    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial features for selected numeric columns.

        Generates squared terms (and optionally higher-order terms) for
        features with suspected non-linear relationships to the target.
        """
        degree = self.config.get("polynomial_degree", 2)
        poly_cols = [c for c in ["age", "bmi", "creatinine", "glucose",
                                  "length_of_stay"] if c in df.columns]

        for col in poly_cols:
            for d in range(2, degree + 1):
                df[f"{col}_pow{d}"] = df[col] ** d

        logger.debug(f"Created polynomial features (degree={degree}) for {len(poly_cols)} columns")
        return df

    def get_feature_names(self, df: pd.DataFrame,
                          exclude_cols: Optional[list[str]] = None) -> list[str]:
        """
        Get the list of feature column names, excluding identifiers and targets.

        Args:
            df: DataFrame to extract feature names from.
            exclude_cols: Additional columns to exclude from features.

        Returns:
            List of feature column names.
        """
        default_exclude = {
            "patient_id", "event_id", "observation_date", "admission_date",
            "discharge_date", "readmission_30day",
        }
        if exclude_cols:
            default_exclude.update(exclude_cols)

        feature_cols = [c for c in df.columns if c not in default_exclude]
        return feature_cols
