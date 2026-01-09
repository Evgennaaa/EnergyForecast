"""
Модуль для предобработки данных перед прогнозированием.
"""

import pandas as pd
import numpy as np
from typing import Optional


def sort_by_date(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Сортирует DataFrame по дате.
    
    Args:
        df: DataFrame
        date_column: Название колонки с датой
    
    Returns:
        Отсортированный DataFrame
    """
    df = df.copy()
    df = df.sort_values(by=date_column).reset_index(drop=True)
    return df


def handle_missing_values(
    df: pd.DataFrame,
    target_column: str,
    method: str = "forward_fill"
) -> pd.DataFrame:
    """
    Обрабатывает пропущенные значения в данных.
    
    Args:
        df: DataFrame
        target_column: Название колонки с целевой переменной
        method: Метод обработки ('forward_fill', 'backward_fill', 'interpolate', 'drop')
    
    Returns:
        DataFrame с обработанными пропущенными значениями
    """
    df = df.copy()
    
    if target_column not in df.columns:
        return df
    
    if method == "forward_fill":
        df[target_column] = df[target_column].ffill()
    elif method == "backward_fill":
        df[target_column] = df[target_column].bfill()
    elif method == "interpolate":
        df[target_column] = df[target_column].interpolate(method="linear")
    elif method == "drop":
        df = df.dropna(subset=[target_column])

    # Если остались пропуски, заполняем средним
    if df[target_column].isna().any():
        df[target_column] = df[target_column].fillna(df[target_column].mean())
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    target_column: str,
    method: str = "iqr",
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Удаляет выбросы из данных.
    
    Args:
        df: DataFrame
        target_column: Название колонки с целевой переменной
        method: Метод определения выбросов ('iqr', 'zscore')
        factor: Коэффициент для метода IQR
    
    Returns:
        DataFrame без выбросов
    """
    df = df.copy()
    
    if target_column not in df.columns:
        return df
    
    if method == "iqr":
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask = (df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)
    elif method == "zscore":
        z_scores = np.abs((df[target_column] - df[target_column].mean()) / df[target_column].std())
        mask = z_scores < 3  # 3 стандартных отклонения
    else:
        return df
    
    return df[mask].reset_index(drop=True)


def prepare_data_for_forecast(
    df: pd.DataFrame,
    date_column: str,
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Подготавливает данные для прогнозирования: сортировка, обработка пропусков.
    
    Args:
        df: DataFrame
        date_column: Название колонки с датой
        target_column: Название колонки с целевой переменной (опционально)
    
    Returns:
        Подготовленный DataFrame
    """
    df = df.copy()
    
    # Сортировка по дате
    df = sort_by_date(df, date_column)
    
    # Обработка пропущенных значений в целевой переменной
    if target_column and target_column in df.columns:
        df = handle_missing_values(df, target_column, method="forward_fill")
    
    return df
