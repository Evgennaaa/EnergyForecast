"""
Модуль для генерации признаков для модели XGBoost.
"""

import pandas as pd
import numpy as np
from typing import Optional


def extract_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Извлекает временные признаки из колонки с датой.
    
    Args:
        df: DataFrame
        date_column: Название колонки с датой/временем
    
    Returns:
        DataFrame с добавленными временными признаками
    """
    df = df.copy()
    
    if date_column not in df.columns:
        return df
    
    # Убеждаемся, что колонка в формате datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Извлечение временных признаков
    df["hour"] = df[date_column].dt.hour
    df["day_of_week"] = df[date_column].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["month"] = df[date_column].dt.month
    df["day_of_year"] = df[date_column].dt.dayofyear
    df["is_weekend"] = (df[date_column].dt.dayofweek >= 5).astype(int)
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 48, 168]
) -> pd.DataFrame:
    """
    Создаёт лаговые признаки (значения целевой переменной в предыдущие моменты времени).
    Использует расширенный набор лагов как в ноутбуке ML_analysis.ipynb.
    
    Args:
        df: DataFrame
        target_column: Название колонки с целевой переменной
        lags: Список лагов (в часах)
    
    Returns:
        DataFrame с добавленными лаговыми признаками
    """
    df = df.copy()
    
    if target_column not in df.columns:
        return df
    
    for lag in lags:
        df[f"lag_{lag}"] = df[target_column].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    windows: list = [24],
    use_shift: bool = True
) -> pd.DataFrame:
    """
    Создаёт скользящие статистики (среднее, стандартное отклонение).
    Использует shift(1) для предотвращения утечки данных (как в ноутбуке).
    
    Args:
        df: DataFrame
        target_column: Название колонки с целевой переменной
        windows: Список размеров окон (в часах)
        use_shift: Использовать ли shift(1) для предотвращения утечки
    
    Returns:
        DataFrame с добавленными скользящими признаками
    """
    df = df.copy()
    
    if target_column not in df.columns:
        return df
    
    for window in windows:
        if use_shift:
            # Без утечки: используем shift(1) перед rolling
            df[f"rolling_mean_{window}"] = df[target_column].shift(1).rolling(window=window, min_periods=1).mean()
            df[f"rolling_std_{window}"] = df[target_column].shift(1).rolling(window=window, min_periods=1).std()
        else:
            df[f"rolling_mean_{window}"] = df[target_column].rolling(window=window, min_periods=1).mean()
            df[f"rolling_std_{window}"] = df[target_column].rolling(window=window, min_periods=1).std()
    
    return df


def engineer_features(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    include_lags: bool = True,
    include_rolling: bool = True
) -> pd.DataFrame:
    """
    Генерирует все признаки для модели XGBoost.
    
    Args:
        df: DataFrame
        date_column: Название колонки с датой/временем
        target_column: Название колонки с целевой переменной
        include_lags: Включать ли лаговые признаки
        include_rolling: Включать ли скользящие признаки
    
    Returns:
        DataFrame с добавленными признаками
    """
    df = df.copy()
    
    # Временные признаки
    df = extract_temporal_features(df, date_column)
    
    # Лаговые признаки (расширенный набор как в ноутбуке)
    if include_lags and target_column in df.columns:
        df = create_lag_features(
            df, target_column, 
            lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 48, 168]
        )
    
    # Скользящие признаки (только 24h как в ноутбуке, с shift для предотвращения утечки)
    if include_rolling and target_column in df.columns:
        df = create_rolling_features(df, target_column, windows=[24], use_shift=True)
    
    # Индекс времени (как в build_features_X_optimized)
    # Добавляем time_idx для соответствия с моделью (34-й признак)
    df["time_idx"] = np.arange(len(df), dtype=np.int32)
    
    # Заполнение пропусков в признаках (для первых строк, где нет лагов)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].bfill().fillna(0)
    
    return df


def prepare_features_for_prediction(
    df: pd.DataFrame,
    feature_columns: list,
    date_column: str,
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Подготавливает признаки для прогнозирования.
    Если целевая переменная отсутствует (для новых данных), использует последние известные значения.
    
    Args:
        df: DataFrame
        feature_columns: Список названий признаков для модели
        date_column: Название колонки с датой
        target_column: Название колонки с целевой переменной (опционально)
    
    Returns:
        DataFrame с подготовленными признаками
    """
    df = df.copy()
    
    # Если целевая переменная есть, генерируем все признаки
    if target_column and target_column in df.columns:
        df = engineer_features(df, date_column, target_column)
    else:
        # Если целевой переменной нет, создаём только временные признаки
        df = extract_temporal_features(df, date_column)
        # Лаговые и скользящие признаки будут недоступны
        # В реальном приложении нужно было бы использовать последние известные значения
    
    # Выбираем только нужные признаки
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        # Заполняем отсутствующие признаки нулями или средними значениями
        for col in missing_features:
            df[col] = 0
    
    return df[available_features + missing_features] if missing_features else df[available_features]

