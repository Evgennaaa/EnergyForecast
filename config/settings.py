"""
Глобальные настройки приложения для прогнозирования энергопотребления.
"""

import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "samples"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "weights"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

# Пути к файлам
# XGBoost модели (отдельные для каждого горизонта)
XGB_MODEL_PATHS = {
    1: MODELS_DIR / "xgb_final_h1.joblib",
    24: MODELS_DIR / "xgb_final_h24.joblib",
    168: MODELS_DIR / "xgb_final_h168.joblib"
}

# Старые пути (для обратной совместимости / тестовой модели)
MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
METADATA_PATH = METADATA_DIR / "xgboost_meta.json"
SAMPLE_DATA_PATH = DATA_DIR / "sample_1000_hours.csv"

# Параметры прогнозирования
FORECAST_HORIZONS = {
    "1 час": 1,
    "24 часа": 24,
    "168 часов (1 неделя)": 168
}

# Параметры временного ряда
DATE_COLUMN = "datetime"
TARGET_COLUMN = "Usage_kWh"  # кВт·ч

# Параметры модели
# Расширенный набор признаков как в ноутбуке ML_analysis.ipynb
FEATURE_COLUMNS = [
    "month",
    "day_of_week",
    "hour",
    "is_weekend",
    "day_of_year",
    # Лаги 1-24, 48, 168
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", "lag_8",
    "lag_9", "lag_10", "lag_11", "lag_12", "lag_13", "lag_14", "lag_15", "lag_16",
    "lag_17", "lag_18", "lag_19", "lag_20", "lag_21", "lag_22", "lag_23", "lag_24",
    "lag_48", "lag_168",
    # Скользящие статистики (только 24h, с shift для предотвращения утечки)
    "rolling_mean_24",
    "rolling_std_24",
    # Индекс времени (как в build_features_X_optimized)
    "time_idx"
]

# Настройки Streamlit
PAGE_TITLE = "Прогнозирование энергопотребления"
PAGE_ICON = ""

# Настройки визуализации
PLOT_HEIGHT = 500
PLOT_THEME = "plotly_white"
