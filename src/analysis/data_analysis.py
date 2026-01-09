"""
Модуль для анализа данных энергопотребления.
Включает функции для построения профилей, декомпозиции и статистического анализа.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def build_daily_profile(df: pd.DataFrame, date_column: str, target_column: str) -> Tuple[pd.Series, pd.Series]:
    """
    Строит суточные профили для рабочих дней и выходных.
    
    Args:
        df: DataFrame с данными
        date_column: Название колонки с датой/временем
        target_column: Название колонки с целевой переменной
    
    Returns:
        Tuple (workday_profile, weekend_profile) - профили для рабочих дней и выходных
    """
    df = df.copy()
    
    # Убеждаемся, что дата в правильном формате
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Устанавливаем индекс на дату для удобства
    if date_column in df.columns:
        df = df.set_index(date_column)
    
    # Извлекаем час и день недели
    df['hour'] = df.index.hour
    df['is_weekend'] = df.index.dayofweek >= 5
    
    # Профили
    workday_profile = df[~df['is_weekend']].groupby('hour')[target_column].mean()
    weekend_profile = df[df['is_weekend']].groupby('hour')[target_column].mean()
    
    return workday_profile, weekend_profile


def build_weekly_profile(df: pd.DataFrame, date_column: str, target_column: str) -> pd.Series:
    """
    Строит недельный профиль (среднее потребление по дням недели).
    
    Args:
        df: DataFrame с данными
        date_column: Название колонки с датой/временем
        target_column: Название колонки с целевой переменной
    
    Returns:
        Series с профилем по дням недели
    """
    df = df.copy()
    
    # Убеждаемся, что дата в правильном формате
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Устанавливаем индекс на дату
    if date_column in df.columns:
        df = df.set_index(date_column)
    
    # Профиль по дням недели
    weekly_profile = df.groupby(df.index.dayofweek)[target_column].mean()
    
    return weekly_profile


def calculate_basic_statistics(df: pd.DataFrame, target_column: str) -> dict:
    """
    Вычисляет базовую статистику по данным.
    
    Args:
        df: DataFrame с данными
        target_column: Название колонки с целевой переменной
    
    Returns:
        Словарь со статистикой
    """
    stats = {
        "count": len(df),
        "mean": df[target_column].mean(),
        "std": df[target_column].std(),
        "min": df[target_column].min(),
        "max": df[target_column].max(),
        "median": df[target_column].median(),
        "q25": df[target_column].quantile(0.25),
        "q75": df[target_column].quantile(0.75),
        "iqr": df[target_column].quantile(0.75) - df[target_column].quantile(0.25)
    }
    
    return stats


def detect_outliers_iqr(df: pd.DataFrame, target_column: str, factor: float = 2.0) -> pd.DataFrame:
    """
    Обнаруживает выбросы методом IQR.
    
    Args:
        df: DataFrame с данными
        target_column: Название колонки с целевой переменной
        factor: Коэффициент для определения границ (по умолчанию 2.0)
    
    Returns:
        DataFrame с выбросами
    """
    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
    return outliers


def get_data_period(df: pd.DataFrame, date_column: str) -> dict:
    """
    Получает информацию о периоде данных.
    
    Args:
        df: DataFrame с данными
        date_column: Название колонки с датой/временем
    
    Returns:
        Словарь с информацией о периоде
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    period_info = {
        "start": df[date_column].min(),
        "end": df[date_column].max(),
        "days": (df[date_column].max() - df[date_column].min()).days,
        "hours": len(df),
        "expected_hours": (df[date_column].max() - df[date_column].min()).total_seconds() / 3600 + 1
    }
    
    return period_info


