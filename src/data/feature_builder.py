"""
Оптимизированная версия build_features_X для минимизации копий данных.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def build_features_X_optimized(
    df: pd.DataFrame,
    target_col: str = "Usage_kWh",
    freq: str = "h",
    lags: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 48, 168),
    rolling_window: int = 24,
    add_time_idx: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Оптимизированная версия build_features_X: минимизирует копии данных.
    
    Строит X(t) на основе прошлого (без утечки) и возвращает:
      X: признаки (DataFrame)
      y: исходный ряд y(t) (Series)
    
    Требования:
      - df.index = datetime
      - df[target_col] существует
      - частота наблюдений = 1 час (или согласуется с freq)
    
    Оптимизации:
      - Создаёт новый DataFrame вместо копирования всего df
      - Использует inplace операции где возможно
      - Минимизирует количество копий
    
    Args:
        df: DataFrame с индексом DateTime и колонкой target_col
        target_col: Название колонки с целевой переменной
        freq: Частота данных ('h' для часовых)
        lags: Кортеж лагов для создания признаков
        rolling_window: Размер окна для скользящих статистик
        add_time_idx: Добавлять ли индекс времени
    
    Returns:
        Tuple (X, y) где:
        - X: DataFrame с признаками (индекс DateTime)
        - y: Series с целевой переменной (индекс DateTime)
    """
    # 0) Порядок и фиксация частоты
    # sort_index может создать копию, но это необходимо для корректной работы
    df_sorted = df.sort_index()
    df_freq = df_sorted.asfreq(freq)  # Может создать копию, но необходимо
    
    # 1) Базовый ряд (view, не копия)
    y = df_freq[target_col].astype(np.float32)
    
    # 2) Создаём новый DataFrame (не копируем данные из df)
    X = pd.DataFrame(index=df_freq.index)
    
    # 3) Календарные признаки (inplace операции на новом DataFrame)
    idx = X.index
    X["month"] = idx.month.astype(np.int16)
    X["day_of_week"] = idx.dayofweek.astype(np.int8)
    X["hour"] = idx.hour.astype(np.int8)
    X["is_weekend"] = idx.dayofweek.isin([5, 6]).astype(np.int8)
    X["day_of_year"] = idx.dayofyear.astype(np.int16)
    
    # 4) Лаги (используем y, не копируем df)
    for lag in lags:
        X[f"lag_{lag}"] = y.shift(lag)
    
    # 5) Rolling без утечки (используем y, не копируем df)
    X[f"rolling_mean_{rolling_window}"] = y.shift(1).rolling(rolling_window, min_periods=1).mean()
    X[f"rolling_std_{rolling_window}"] = y.shift(1).rolling(rolling_window, min_periods=1).std()
    
    # 6) Индекс времени (опционально)
    if add_time_idx:
        X["time_idx"] = np.arange(len(X), dtype=np.int32)
    
    # 7) Удаляем строки, где фичи не посчитались (из-за лагов/rolling)
    # Это создаёт новые объекты, но необходимо для корректности
    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]
    
    # 8) Оптимизация типов (inplace для экономии памяти)
    float_cols = X.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        X[float_cols] = X[float_cols].astype(np.float32)
    
    return X, y


def build_targets_multi_horizon(
    y: pd.Series,
    horizons: tuple = (1, 24, 168)
) -> dict:
    """
    Создаёт таргеты y(t+h) для нескольких горизонтов.
    
    Args:
        y: Series с целевой переменной
        horizons: Кортеж горизонтов прогнозирования
    
    Returns:
        Словарь {h: {"y": Series, "mask": Series[bool]}}
    """
    targets = {}
    
    for h in horizons:
        # y(t+h) = shift(-h)
        y_h = y.shift(-h)
        
        # Маска: где есть и X, и y(t+h)
        mask = y.notna() & y_h.notna()
        
        targets[h] = {
            "y": y_h,
            "mask": mask
        }
    
    return targets


def time_split_Xy(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Разделяет данные по времени на train/val/test.
    
    Args:
        X: DataFrame с признаками
        y: Series с целевой переменной
        train_ratio: Доля обучающей выборки
        val_ratio: Доля валидационной выборки
    
    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

