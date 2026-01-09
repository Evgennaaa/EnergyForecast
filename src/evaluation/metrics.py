"""
Модуль для вычисления метрик оценки качества прогноза.
"""

import numpy as np
import pandas as pd
from typing import Optional


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную ошибку (MAE).
    
    Args:
        y_true: Истинные значения
        y_pred: Прогнозируемые значения
    
    Returns:
        MAE
    """
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет корень из средней квадратичной ошибки (RMSE).
    
    Args:
        y_true: Истинные значения
        y_pred: Прогнозируемые значения
    
    Returns:
        RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Вычисляет симметричную среднюю абсолютную процентную ошибку (sMAPE).
    
    Args:
        y_true: Истинные значения
        y_pred: Прогнозируемые значения
    
    Returns:
        sMAPE в процентах
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Избегаем деления на ноль
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0
    smape = np.mean(numerator[mask] / denominator[mask]) * 100
    return smape


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None
) -> float:
    """
    Вычисляет среднюю абсолютную масштабированную ошибку (MASE).
    
    Args:
        y_true: Истинные значения
        y_pred: Прогнозируемые значения
        y_train: Обучающие данные для вычисления масштабирующего фактора
    
    Returns:
        MASE
    """
    if y_train is None:
        # Если обучающие данные не предоставлены, используем naive forecast
        # (прогноз = предыдущее значение)
        if len(y_true) < 2:
            return np.nan
        naive_forecast = np.roll(y_true, 1)
        naive_forecast[0] = y_true[0]  # Первое значение остаётся как есть
        mae_naive = mean_absolute_error(y_true, naive_forecast)
    else:
        # Используем сезонный naive forecast (значение 24 часа назад)
        if len(y_train) < 24:
            return np.nan
        seasonal_naive = np.roll(y_train, 24)
        mae_naive = mean_absolute_error(y_train[24:], seasonal_naive[24:])
    
    if mae_naive == 0:
        return np.nan
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae / mae_naive


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None
) -> dict:
    """
    Вычисляет все метрики оценки качества прогноза.
    
    Args:
        y_true: Истинные значения
        y_pred: Прогнозируемые значения
        y_train: Обучающие данные (для MASE)
    
    Returns:
        Словарь с метриками
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "sMAPE": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "MASE": mean_absolute_scaled_error(y_true, y_pred, y_train)
    }
    
    return metrics


def format_metrics(metrics: dict) -> pd.DataFrame:
    """
    Форматирует метрики в виде DataFrame для отображения.
    
    Args:
        metrics: Словарь с метриками
    
    Returns:
        DataFrame с метриками
    """
    df = pd.DataFrame([
        {"Метрика": "MAE (кВт·ч)", "Значение": f"{metrics['MAE']:.2f}"},
        {"Метрика": "RMSE (кВт·ч)", "Значение": f"{metrics['RMSE']:.2f}"},
        {"Метрика": "sMAPE (%)", "Значение": f"{metrics['sMAPE']:.2f}"},
        {"Метрика": "MASE", "Значение": f"{metrics['MASE']:.4f}" if not np.isnan(metrics['MASE']) else "N/A"},
    ])
    return df


