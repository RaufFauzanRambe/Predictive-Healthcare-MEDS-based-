"""
preprocess.py - Healthcare data preprocessing pipeline.

Handles missing value imputation, outlier treatment, normalization,
categorical encoding, and time-series specific preprocessing such as
resampling, interpolation, and sequence windowing.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    QuantileTransformer,
    TargetEncoder,
)

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class Preprocessor:
    """
    Comprehensive preprocessing pipeline for healthcare data.

    Supports configurable strategies for missing values, outlier handling,
    normalization, and categorical encoding. Includes specialized time-series
    preprocessing for longitudinal patient data.

    Attributes:
        config: Preprocessing configuration dictionary.
        fitted: Whether the preprocessor has been fitted on training data.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Initialize the Preprocessor.

        Args:
            config: Preprocessing configuration dict. If None, loads from
                    data_config.yaml under the 'preprocessing' key.
        """
        if config is None:
            data_config = load_config("configs/data_config.yaml")
            config = data_config.get("preprocessing", {})

        self.config = config
        self.fitted = False
        self._imputers: dict[str, Any] = {}
        self._scalers: dict[str, Any] = {}
        self._encoders: dict[str, Any] = {}
        self._feature_names: list[str] = []

    def fit_transform(self, df: pd.DataFrame, columns_config: Optional[dict] = None) -> pd.DataFrame:
        """
        Fit the preprocessing pipeline on the data and transform it.

        This is the primary entry point. It sequentially applies:
        1. Missing value imputation
        2. Outlier treatment
        3. Categorical encoding
        4. Numerical normalization
        5. Time-series preprocessing (if applicable)

        Args:
            df: Raw input DataFrame.
            columns_config: Column type definitions (numeric, categorical, binary).

        Returns:
            Preprocessed DataFrame ready for feature engineering or modeling.
        """
        logger.info(f"Fitting preprocessor on {len(df)} records, {len(df.columns)} columns")

        if columns_config is None:
            data_config = load_config("configs/data_config.yaml")
            columns_config = data_config.get("columns", {})

        df = df.copy()

        # Step 1: Drop columns with excessive missing values
        df = self._drop_high_missing(df)

        # Step 2: Impute missing values
        df = self._impute_missing(df, columns_config)

        # Step 3: Handle outliers
        df = self._handle_outliers(df, columns_config)

        # Step 4: Encode categorical features
        df = self._encode_categorical(df, columns_config)

        # Step 5: Normalize numeric features
        df = self._normalize(df, columns_config)

        self.fitted = True
        self._feature_names = list(df.columns)
        logger.info(f"Preprocessing complete. Output shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame, columns_config: Optional[dict] = None) -> pd.DataFrame:
        """
        Transform new data using the already-fitted pipeline.

        Uses the imputers, scalers, and encoders learned during fit_transform.
        Should only be called after fit_transform has been called.

        Args:
            df: New data to transform.
            columns_config: Column type definitions.

        Returns:
            Preprocessed DataFrame.

        Raises:
            RuntimeError: If called before fit_transform.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform(). "
                               "Call fit_transform() first.")

        if columns_config is None:
            data_config = load_config("configs/data_config.yaml")
            columns_config = data_config.get("columns", {})

        df = df.copy()
        df = self._impute_missing(df, columns_config, use_fitted=True)
        df = self._handle_outliers(df, columns_config)
        df = self._encode_categorical(df, columns_config, use_fitted=True)
        df = self._normalize(df, columns_config, use_fitted=True)
        return df

    def _drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns where the missing ratio exceeds the configured threshold."""
        max_ratio = self.config.get("missing_values", {}).get("max_missing_ratio", 0.8)
        missing_ratios = df.isnull().mean()
        cols_to_drop = missing_ratios[missing_ratios > max_ratio].index.tolist()

        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{max_ratio:.0%} missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        return df

    def _impute_missing(self, df: pd.DataFrame, columns_config: dict,
                        use_fitted: bool = False) -> pd.DataFrame:
        """
        Impute missing values using the configured strategy.

        Supports: mean, median, mode, knn, and iterative (MICE) imputation.
        Adds binary indicator features for missingness when configured.
        """
        cfg = self.config.get("missing_values", {})
        strategy = cfg.get("strategy", "median")
        add_indicators = cfg.get("indicator_features", True)

        numeric_cols = [c for c in columns_config.get("numeric", []) if c in df.columns]
        categorical_cols = [c for c in columns_config.get("categorical", []) if c in df.columns]

        # Add missing indicators before imputation
        if add_indicators:
            for col in numeric_cols + categorical_cols:
                if df[col].isnull().any():
                    df[f"{col}_missing"] = df[col].isnull().astype(int)

        # Impute numeric columns
        if numeric_cols and df[numeric_cols].isnull().any().any():
            if use_fitted and "numeric" in self._imputers:
                imputer = self._imputers["numeric"]
            else:
                if strategy == "knn":
                    imputer = KNNImputer(n_neighbors=cfg.get("knn_neighbors", 5))
                elif strategy == "iterative":
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                else:
                    imputer = SimpleImputer(strategy=strategy)

                imputer.fit(df[numeric_cols])
                self._imputers["numeric"] = imputer

            df[numeric_cols] = imputer.transform(df[numeric_cols])

        # Impute categorical columns with mode
        if categorical_cols and df[categorical_cols].isnull().any().any():
            if use_fitted and "categorical" in self._imputers:
                imputer = self._imputers["categorical"]
            else:
                imputer = SimpleImputer(strategy="most_frequent")
                imputer.fit(df[categorical_cols])
                self._imputers["categorical"] = imputer

            df[categorical_cols] = imputer.transform(df[categorical_cols])

        return df

    def _handle_outliers(self, df: pd.DataFrame, columns_config: dict) -> pd.DataFrame:
        """
        Detect and handle outliers in numeric columns.

        Supports IQR, Z-score, and Winsorize methods. Outliers can be
        clipped, removed, or flagged with a binary indicator column.
        """
        cfg = self.config.get("outliers", {})
        method = cfg.get("method", "iqr")
        action = cfg.get("action", "clip")
        numeric_cols = [c for c in columns_config.get("numeric", []) if c in df.columns]

        for col in numeric_cols:
            if method == "iqr":
                factor = cfg.get("iqr_factor", 1.5)
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
            elif method == "zscore":
                threshold = cfg.get("zscore_threshold", 3.0)
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
            elif method == "winsorize":
                limits = cfg.get("winsorize_limits", [0.01, 0.99])
                lower = df[col].quantile(limits[0])
                upper = df[col].quantile(limits[1])
            else:
                continue

            outlier_mask = (df[col] < lower) | (df[col] > upper)
            n_outliers = outlier_mask.sum()

            if n_outliers > 0:
                logger.debug(f"Found {n_outliers} outliers in '{col}' using {method}")

                if action == "clip":
                    df[col] = df[col].clip(lower=lower, upper=upper)
                elif action == "flag":
                    df[f"{col}_outlier"] = outlier_mask.astype(int)
                elif action == "remove":
                    df = df[~outlier_mask]

        return df

    def _encode_categorical(self, df: pd.DataFrame, columns_config: dict,
                            use_fitted: bool = False) -> pd.DataFrame:
        """
        Encode categorical columns using the configured method.

        Supports: onehot, label, target, and frequency encoding.
        Target encoding requires the target column to be present in the DataFrame.
        """
        cfg = self.config.get("encoding", {})
        method = cfg.get("categorical_method", "onehot")
        max_categories = cfg.get("max_categories", 20)
        categorical_cols = [c for c in columns_config.get("categorical", []) if c in df.columns]

        if not categorical_cols:
            return df

        for col in categorical_cols:
            n_unique = df[col].nunique()

            if method == "onehot" and n_unique <= max_categories:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            elif method == "label":
                if use_fitted and col in self._encoders:
                    le = self._encoders[col]
                    df[col] = df[col].map(
                        lambda x, le=le: le.transform([x])[0]
                        if x in le.classes_ else -1
                    )
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._encoders[col] = le
            elif method == "frequency":
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map).fillna(0)
            elif method == "target":
                # Target encoding — uses the target column if available
                target_col = "readmission_30day"
                if target_col in df.columns:
                    target_mean = df.groupby(col)[target_col].mean()
                    df[col] = df[col].map(target_mean).fillna(df[target_col].mean())
                else:
                    # Fallback to frequency encoding if target is not available
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0)

        return df

    def _normalize(self, df: pd.DataFrame, columns_config: dict,
                   use_fitted: bool = False) -> pd.DataFrame:
        """
        Normalize numeric features using the configured method.

        Supports: standard (z-score), minmax, robust, and quantile scaling.
        Scalers are fitted on training data and reused for validation/test.
        """
        cfg = self.config.get("normalization", {})
        method = cfg.get("method", "standard")
        numeric_cols = [c for c in columns_config.get("numeric", []) if c in df.columns]

        if not numeric_cols:
            return df

        if use_fitted and "scaler" in self._scalers:
            scaler = self._scalers["scaler"]
            df[numeric_cols] = scaler.transform(df[numeric_cols])
            return df

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "quantile":
            scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        else:
            logger.warning(f"Unknown normalization method '{method}', skipping")
            return df

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self._scalers["scaler"] = scaler
        return df

    def preprocess_time_series(
        self,
        df: pd.DataFrame,
        patient_id_col: str = "patient_id",
        timestamp_col: str = "observation_date",
    ) -> pd.DataFrame:
        """
        Preprocess healthcare time-series data for sequence modeling.

        Handles resampling to regular intervals, interpolation of missing
        values within each patient's timeline, and creates fixed-length
        sequence windows for LSTM/Transformer input.

        Args:
            df: DataFrame with time-indexed patient observations.
            patient_id_col: Column name for patient identifiers.
            timestamp_col: Column name for observation timestamps.

        Returns:
            DataFrame with resampled and interpolated time-series per patient.
        """
        ts_cfg = self.config.get("time_series", {})
        resample_freq = ts_cfg.get("resample_freq", "1H")
        interpolation = ts_cfg.get("interpolation", "linear")

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([patient_id_col, timestamp_col])

        resampled_frames: list[pd.DataFrame] = []
        numeric_agg = ts_cfg.get("aggregation", {}).get("numeric", "mean")

        for pid, group in df.groupby(patient_id_col):
            group = group.set_index(timestamp_col)

            # Resample to regular frequency
            group = group.resample(resample_freq).agg(
                {col: numeric_agg for col in group.select_dtypes(include=[np.number]).columns}
            )

            # Interpolate missing time steps
            group = group.interpolate(method=interpolation, limit_direction="both")

            # Forward/backward fill any remaining gaps
            group = group.ffill().bfill()

            group[patient_id_col] = pid
            resampled_frames.append(group)

        result = pd.concat(resampled_frames, ignore_index=False).reset_index()
        logger.info(f"Time-series preprocessing: {len(df)} → {len(result)} records after resampling")
        return result

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "readmission_30day",
        patient_id_col: str = "patient_id",
        lookback: int = 48,
        horizon: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create fixed-length sequence windows from patient time-series data.

        Generates (lookback) hour input sequences and corresponding targets
        for sequence models (LSTM, Transformer).

        Args:
            df: Time-series DataFrame sorted by patient and timestamp.
            feature_cols: List of feature column names.
            target_col: Target variable column name.
            patient_id_col: Patient identifier column.
            lookback: Number of time steps for input sequence.
            horizon: Number of time steps ahead for the target.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, lookback, n_features)
            and y has shape (n_samples,).
        """
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for pid, group in df.groupby(patient_id_col):
            features = group[feature_cols].values.astype(np.float32)
            targets = group[target_col].values.astype(np.float32)

            if len(features) < lookback + horizon:
                continue

            for i in range(len(features) - lookback - horizon + 1):
                X_list.append(features[i : i + lookback])
                # Target is the value at horizon steps after the lookback window
                y_list.append(targets[i + lookback + horizon - 1])

        X = np.stack(X_list)
        y = np.stack(y_list)
        logger.info(f"Created sequences: X={X.shape}, y={y.shape}, "
                     f"positive rate={y.mean():.3f}")
        return X, y
