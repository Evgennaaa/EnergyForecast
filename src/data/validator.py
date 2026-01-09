"""
Модуль для валидации данных перед прогнозированием.
"""

import pandas as pd
from typing import List, Tuple, Optional
import streamlit as st


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    date_column: str,
    target_column: str
) -> Tuple[bool, Optional[str]]:
    """
    Валидирует DataFrame на наличие необходимых колонок и корректность данных.
    
    Args:
        df: DataFrame для валидации
        required_columns: Список обязательных колонок
        date_column: Название колонки с датой/временем
        target_column: Название колонки с целевой переменной
    
    Returns:
        Tuple (is_valid, error_message)
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return False, "DataFrame пуст"
    
    # Проверка наличия обязательных колонок
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Отсутствуют обязательные колонки: {', '.join(missing_columns)}"
    
    # Проверка колонки с датой
    if date_column not in df.columns:
        return False, f"Колонка с датой '{date_column}' не найдена"
    
    # Проверка, что дата в правильном формате
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        return False, f"Колонка '{date_column}' должна быть типа datetime"
    
    # Проверка наличия целевой переменной (если она нужна для обучения/валидации)
    if target_column and target_column not in df.columns:
        return False, f"Колонка с целевой переменной '{target_column}' не найдена"
    
    # Проверка на пропущенные значения в дате
    if df[date_column].isna().any():
        return False, f"Обнаружены пропущенные значения в колонке '{date_column}'"
    
    # Проверка на дубликаты по дате
    if df[date_column].duplicated().any():
        return False, f"Обнаружены дубликаты в колонке '{date_column}'"
    
    # Проверка на сортировку по дате
    if not df[date_column].is_monotonic_increasing:
        st.warning("️ Данные не отсортированы по дате. Рекомендуется отсортировать.")
    
    return True, None


def validate_forecast_horizon(horizon: int, max_horizon: int = 168) -> Tuple[bool, Optional[str]]:
    """
    Валидирует горизонт прогнозирования.
    
    Args:
        horizon: Горизонт прогнозирования (количество часов)
        max_horizon: Максимально допустимый горизонт
    
    Returns:
        Tuple (is_valid, error_message)
    """
    if horizon <= 0:
        return False, "Горизонт прогнозирования должен быть положительным числом"
    
    if horizon > max_horizon:
        return False, f"Горизонт прогнозирования не должен превышать {max_horizon} часов"
    
    return True, None


def check_data_quality(df: pd.DataFrame, target_column: str) -> dict:
    """
    Проверяет качество данных и возвращает статистику.
    
    Args:
        df: DataFrame
        target_column: Название колонки с целевой переменной
    
    Returns:
        Словарь со статистикой качества данных
    """
    stats = {
        "total_rows": len(df),
        "missing_values": df[target_column].isna().sum() if target_column in df.columns else 0,
        "negative_values": (df[target_column] < 0).sum() if target_column in df.columns else 0,
        "zero_values": (df[target_column] == 0).sum() if target_column in df.columns else 0,
    }
    
    if target_column in df.columns:
        stats["mean"] = df[target_column].mean()
        stats["std"] = df[target_column].std()
        stats["min"] = df[target_column].min()
        stats["max"] = df[target_column].max()
    
    return stats


