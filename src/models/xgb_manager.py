"""
XGBoost model manager for loading and forecasting.
Supports one model per horizon.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
import streamlit as st


class XGBModelManager:
    """Manage XGBoost models (one per horizon)."""

    def __init__(self, model_paths: Dict[int, Path], metadata_path: Optional[Path] = None):
        """
        Initialize XGBModelManager.

        Args:
            model_paths: Dict {horizon: path} for each horizon model
            metadata_path: Optional path to metadata file
        """
        self.model_paths = model_paths
        self.metadata_path = metadata_path
        self.models: Dict[int, Any] = {}
        self.metadata = None
        self.feature_names = None

    def load_model(self, horizon: int) -> bool:
        """
        Load model for the given horizon.

        Args:
            horizon: Forecast horizon (1, 24, or 168)

        Returns:
            True if loaded, False otherwise
        """
        if horizon not in self.model_paths:
            st.error(f"Unsupported horizon: {horizon}")
            return False

        model_path = self.model_paths[horizon]

        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return False

        try:
            self.models[horizon] = joblib.load(model_path)
            return True
        except Exception as e:
            st.error(f"Failed to load model for horizon {horizon}: {str(e)}")
            return False

    def load_all_models(self) -> bool:
        """
        Load all horizon models.

        Returns:
            True if all models are loaded, False otherwise
        """
        missing = [h for h, path in self.model_paths.items() if not path.exists()]
        if missing:
            missing_paths = [str(self.model_paths[h]) for h in missing]
            st.error("Missing XGBoost models for required horizons.")
            st.info("Expected files:\n" + "\n".join(missing_paths))
            return False

        for horizon in self.model_paths.keys():
            if not self.load_model(horizon):
                return False

        return True

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load model metadata (params, feature names).

        Returns:
            Metadata dict or None
        """
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                if "feature_names" in self.metadata:
                    self.feature_names = self.metadata["feature_names"]

                return self.metadata
            except Exception as e:
                st.warning(f"Failed to load metadata: {str(e)}")
                return None
        return None

    def predict(self, features: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Predict using the model for the given horizon.

        Args:
            features: Feature DataFrame
            horizon: Forecast horizon (1, 24, or 168)

        Returns:
            Prediction array
        """
        if horizon not in self.models:
            raise ValueError(
                f"Model for horizon {horizon} is not loaded. Call load_model({horizon}) first."
            )

        try:
            model = self.models[horizon]
            predictions = model.predict(features)
            predictions = np.maximum(predictions, 0)
            return predictions
        except Exception as e:
            st.error(f"Prediction failed for horizon {horizon}: {str(e)}")
            raise

    def predict_forecast(
        self,
        historical_data: pd.DataFrame,
        horizon: int,
        feature_columns: list,
        date_column: str,
        target_column: str
    ) -> pd.DataFrame:
        """
        Forecast for the selected horizon using direct strategy.
        Each horizon uses its own model to predict value at t+h.

        Args:
            historical_data: Historical data to initialize the forecast
            horizon: Forecast horizon in hours
            feature_columns: Model feature names
            date_column: Datetime column name
            target_column: Target column name

        Returns:
            DataFrame with datetime and forecast
        """
        from src.features.feature_engineer import engineer_features

        if horizon not in self.models:
            raise ValueError(f"Model for horizon {horizon} is not loaded.")

        df = historical_data.copy()
        last_date = df[date_column].iloc[-1]
        forecast_date = last_date + pd.Timedelta(hours=horizon)

        temp_df = engineer_features(df, date_column, target_column)

        available_features = [col for col in feature_columns if col in temp_df.columns]
        missing_features = [col for col in feature_columns if col not in temp_df.columns]
        for col in missing_features:
            temp_df[col] = 0

        features = temp_df[available_features + missing_features].iloc[[-1]]

        if self.feature_names:
            missing_in_features = set(self.feature_names) - set(features.columns)
            for col in missing_in_features:
                features[col] = 0
            features = features[self.feature_names]

        prediction = self.predict(features, horizon)[0]

        return pd.DataFrame(
            [{date_column: forecast_date, "forecast": float(prediction)}]
        )

    def predict_rolling_direct_series(
        self,
        historical_data: pd.DataFrame,
        horizon: int,
        n_points: int,
        feature_columns: list,
        date_column: str,
        target_column: str
    ) -> pd.DataFrame:
        """
        Build a forecast series using a direct model for each origin time.
        For the last n_points timestamps t, predict the value at t + horizon.

        Args:
            historical_data: Historical data to initialize the forecast
            horizon: Forecast horizon in hours
            n_points: Number of forecast points to generate
            feature_columns: Model feature names
            date_column: Datetime column name
            target_column: Target column name

        Returns:
            DataFrame with datetime and forecast series
        """
        from src.features.feature_engineer import engineer_features

        horizon = int(horizon)
        n_points = int(n_points)

        if horizon not in self.models:
            raise ValueError(f"Model for horizon {horizon} is not loaded.")
        if n_points <= 0:
            raise ValueError("n_points must be a positive integer.")

        df = historical_data.copy()
        df = df.sort_values(date_column)

        max_lag = 0
        for col in feature_columns:
            if col.startswith("lag_"):
                try:
                    max_lag = max(max_lag, int(col.split("_", 1)[1]))
                except ValueError:
                    continue
        min_required = max_lag + n_points
        if len(df) < min_required:
            raise ValueError(
                "Not enough history for rolling direct forecast: "
                f"need at least {min_required} rows, got {len(df)}."
            )

        feat_df = engineer_features(df, date_column, target_column)
        origins = feat_df.tail(n_points).copy()

        available_features = [col for col in feature_columns if col in origins.columns]
        missing_features = [col for col in feature_columns if col not in origins.columns]
        for col in missing_features:
            origins[col] = 0
        features = origins[available_features + missing_features]

        if self.feature_names:
            missing_in_features = set(self.feature_names) - set(features.columns)
            for col in missing_in_features:
                features[col] = 0
            features = features[self.feature_names]

        predictions = self.predict(features, horizon)
        origin_times = pd.to_datetime(origins[date_column])
        forecast_times = origin_times + pd.Timedelta(hours=horizon)

        return pd.DataFrame(
            {date_column: forecast_times, "forecast": predictions}
        )

    def predict_recursive_series(
        self,
        historical_data: pd.DataFrame,
        horizon: int,
        feature_columns: list,
        date_column: str,
        target_column: str
    ) -> pd.DataFrame:
        """
        Build a multi-step forecast series using recursive one-step predictions.

        Args:
            historical_data: Historical data to initialize the forecast
            horizon: Forecast horizon in hours
            feature_columns: Model feature names
            date_column: Datetime column name
            target_column: Target column name

        Returns:
            DataFrame with datetime and forecast series
        """
        from src.features.feature_engineer import engineer_features

        if 1 not in self.models:
            raise ValueError("Model for horizon 1 is not loaded.")

        df = historical_data.copy()
        last_date = df[date_column].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=horizon,
            freq="H"
        )

        forecasts = []

        for forecast_date in forecast_dates:
            temp_df = engineer_features(df, date_column, target_column)

            if "time_idx" in temp_df.columns:
                temp_df["time_idx"] = np.arange(len(temp_df), dtype=np.int32)

            available_features = [col for col in feature_columns if col in temp_df.columns]
            missing_features = [col for col in feature_columns if col not in temp_df.columns]
            for col in missing_features:
                temp_df[col] = 0

            features = temp_df[available_features + missing_features].iloc[[-1]]

            if self.feature_names:
                missing_in_features = set(self.feature_names) - set(features.columns)
                for col in missing_in_features:
                    features[col] = 0
                features = features[self.feature_names]

            prediction = self.predict(features, 1)[0]

            new_row = pd.DataFrame({
                date_column: [forecast_date],
                target_column: [prediction]
            })
            df = pd.concat([df, new_row], ignore_index=True)

            forecasts.append({
                date_column: forecast_date,
                "forecast": float(prediction)
            })

        return pd.DataFrame(forecasts)
