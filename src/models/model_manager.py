"""
Модуль для управления моделью XGBoost: загрузка и прогнозирование.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
import streamlit as st


class ModelManager:
    """Класс для управления моделью XGBoost."""
    
    def __init__(self, model_path: Path, metadata_path: Optional[Path] = None):
        """
        Инициализирует ModelManager.
        
        Args:
            model_path: Путь к файлу модели (.pkl)
            metadata_path: Путь к файлу метаданных (опционально)
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
    
    def load_model(self) -> bool:
        """
        Загружает модель из файла.
        
        Returns:
            True если модель успешно загружена, False иначе
        """
        try:
            if not self.model_path.exists():
                st.error(f"Файл модели не найден: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            st.success(" Модель успешно загружена")
            return True
        except Exception as e:
            st.error(f"Ошибка при загрузке модели: {str(e)}")
            return False
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Загружает метаданные модели (параметры, метрики).
        
        Returns:
            Словарь с метаданными или None
        """
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                return self.metadata
            except Exception as e:
                st.warning(f"Не удалось загрузить метаданные: {str(e)}")
                return None
        return None
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Выполняет прогноз на основе признаков.
        
        Args:
            features: DataFrame с признаками для прогнозирования
        
        Returns:
            Массив с прогнозами
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Вызовите load_model() сначала.")
        
        try:
            predictions = self.model.predict(features)
            # Убеждаемся, что прогнозы неотрицательны (энергопотребление >= 0)
            predictions = np.maximum(predictions, 0)
            return predictions
        except Exception as e:
            st.error(f"Ошибка при прогнозировании: {str(e)}")
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
        Выполняет многошаговый прогноз на заданный горизонт.
        
        Args:
            historical_data: Исторические данные для инициализации прогноза
            horizon: Горизонт прогнозирования (количество часов)
            feature_columns: Список признаков модели
            date_column: Название колонки с датой
            target_column: Название колонки с целевой переменной
        
        Returns:
            DataFrame с прогнозами (datetime и прогноз)
        """
        from src.features.feature_engineer import engineer_features
        
        if self.model is None:
            raise ValueError("Модель не загружена.")
        
        # Копируем исторические данные
        df = historical_data.copy()
        
        # Получаем последнюю дату
        last_date = df[date_column].iloc[-1]
        
        # Генерируем даты для прогноза
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=horizon,
            freq="H"
        )
        
        # Инициализируем список прогнозов
        forecasts = []
        
        # Выполняем пошаговый прогноз
        for i, forecast_date in enumerate(forecast_dates):
            # Генерируем признаки для текущего момента
            temp_df = df.copy()
            temp_df = engineer_features(temp_df, date_column, target_column)
            
            # Берём последнюю строку с признаками
            features = temp_df[feature_columns].iloc[[-1]]
            
            # Выполняем прогноз
            prediction = self.predict(features)[0]
            
            # Добавляем прогноз в историю для следующего шага
            new_row = pd.DataFrame({
                date_column: [forecast_date],
                target_column: [prediction]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            
            forecasts.append({
                date_column: forecast_date,
                "forecast": prediction
            })
        
        return pd.DataFrame(forecasts)


