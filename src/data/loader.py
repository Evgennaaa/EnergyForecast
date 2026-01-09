"""
Модуль для загрузки данных из CSV файлов.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import streamlit as st


def load_csv(file_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Загружает CSV файл в DataFrame.
    
    Args:
        file_path: Путь к CSV файлу
        encoding: Кодировка файла (по умолчанию utf-8)
    
    Returns:
        DataFrame с загруженными данными
    
    Raises:
        FileNotFoundError: Если файл не найден
        pd.errors.EmptyDataError: Если файл пуст
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        if df.empty:
            raise pd.errors.EmptyDataError(f"Файл пуст: {file_path}")
        return df
    except UnicodeDecodeError:
        # Попытка с другой кодировкой
        return pd.read_csv(file_path, encoding="latin-1")


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Загружает файл, загруженный через Streamlit file_uploader.
    
    Args:
        uploaded_file: Объект UploadedFile из Streamlit
    
    Returns:
        DataFrame с загруженными данными
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {str(e)}")
        raise


def ensure_datetime_column(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Преобразует указанную колонку в datetime, если она ещё не в этом формате.
    
    Args:
        df: DataFrame
        date_column: Название колонки с датой/временем
    
    Returns:
        DataFrame с преобразованной колонкой datetime
    """
    df = df.copy()
    if date_column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    return df


def normalize_data_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует данные в стандартный формат приложения.
    Поддерживает два формата:
    1. Стандартный: datetime, Usage_kWh
    2. Альтернативный: Date, Time, Usage_kWh
    
    Args:
        df: DataFrame с данными в любом поддерживаемом формате
    
    Returns:
        DataFrame в стандартном формате (datetime, Usage_kWh)
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Нормализуем имена колонок
    lower_map = {col.lower(): col for col in df.columns}

    datetime_col = None
    for alias in ("datetime", "date_time", "timestamp"):
        if alias in lower_map:
            datetime_col = lower_map[alias]
            break

    usage_col = None
    for alias in ("usage_kwh", "usage", "energy_consumption", "energy", "consumption", "aep_mw", "aepmw"):
        if alias in lower_map:
            usage_col = lower_map[alias]
            break

    rename_map = {}
    if datetime_col and datetime_col != "datetime":
        rename_map[datetime_col] = "datetime"
    if usage_col and usage_col != "Usage_kWh":
        rename_map[usage_col] = "Usage_kWh"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Пересчитываем после переименования
    lower_map = {col.lower(): col for col in df.columns}
    date_col = lower_map.get("date")
    time_col = lower_map.get("time")

    # Проверяем, какой формат данных
    has_datetime = "datetime" in df.columns
    has_date_time = date_col is not None and time_col is not None
    has_usage_kwh = "Usage_kWh" in df.columns
    
    # Если уже в стандартном формате, возвращаем как есть
    if has_datetime and has_usage_kwh:
        return df
    
    # Если формат с Date, Time, Usage_kWh - преобразуем
    if has_date_time and has_usage_kwh:
        # Объединяем Date и Time в datetime
        # Time может быть в формате "00 - 01" или "00:00:00"
        def parse_time(time_str):
            """Извлекает час из строки времени."""
            if pd.isna(time_str):
                return "00"
            
            time_str = str(time_str).strip()
            
            # Формат "00 - 01" или "00-01"
            if " - " in time_str or "-" in time_str:
                # Берём первый час из диапазона
                hour = time_str.split("-")[0].strip()
                return hour.zfill(2)
            
            # Формат "00:00:00" или "00:00"
            if ":" in time_str:
                return time_str.split(":")[0].zfill(2)
            
            # Если просто число
            try:
                hour = int(float(time_str))
                return str(hour).zfill(2)
            except:
                return "00"
        
        # Извлекаем час из Time
        df["hour_str"] = df[time_col].apply(parse_time)
        
        # Объединяем Date и hour_str в datetime
        df["datetime"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df["hour_str"] + ":00:00",
            errors="coerce"
        )
        
        # Оставляем Usage_kWh как есть (не переименовываем)
        # Удаляем временные колонки
        df = df.drop(columns=[date_col, time_col, "hour_str"], errors="ignore")
        
        # Удаляем строки с некорректными датами
        df = df.dropna(subset=["datetime"])
        
        return df
    
    # Если есть только datetime, но нет Usage_kWh - ищем альтернативные названия
    if has_datetime:
        # Пробуем найти колонку с энергопотреблением
        possible_names = ["usage_kwh", "energy_consumption", "energy", "consumption", "aep_mw", "aepmw"]
        for name in possible_names:
            if name in lower_map and lower_map[name] in df.columns:
                df["Usage_kWh"] = df[lower_map[name]]
                break
    
    # Если есть только Date и Usage_kWh (без Time)
    if date_col and "Usage_kWh" in df.columns and not time_col:
        # Предполагаем, что это дневные данные или нужно добавить время
        df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
        # Usage_kWh уже есть, не переименовываем
        df = df.drop(columns=[date_col], errors="ignore")
        df = df.dropna(subset=["datetime"])
        return df
    
    return df
