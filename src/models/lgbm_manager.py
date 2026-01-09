"""
Модуль для управления моделями LightGBM: загрузка и прогнозирование.
Поддерживает отдельные модели для каждого горизонта прогнозирования.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
import streamlit as st
from lightgbm import LGBMRegressor


class LGBMModelManager:
    """Класс для управления моделями LightGBM (отдельные модели для каждого горизонта)."""
    
    def __init__(self, model_paths: Dict[int, Path], metadata_path: Optional[Path] = None):
        """
        Инициализирует LGBMModelManager.
        
        Args:
            model_paths: Словарь {horizon: path} с путями к моделям для каждого горизонта
            metadata_path: Путь к файлу метаданных (опционально)
        """
        self.model_paths = model_paths
        self.metadata_path = metadata_path
        self.models: Dict[int, LGBMRegressor] = {}
        self.metadata = None
        self.feature_names = None
    
    def load_model(self, horizon: int) -> bool:
        """
        Загружает модель для указанного горизонта.
        
        Args:
            horizon: Горизонт прогнозирования (1, 24, или 168)
        
        Returns:
            True если модель успешно загружена, False иначе
        """
        if horizon not in self.model_paths:
            st.error(f"Горизонт {horizon} не поддерживается")
            return False
        
        model_path = self.model_paths[horizon]
        
        try:
            if not model_path.exists():
                # Показываем более информативное сообщение
                st.error(f"Файл модели не найден: {model_path}")
                st.info(f"Убедитесь, что модель для горизонта {horizon} находится в: {model_path.parent}")
                return False
            
            self.models[horizon] = joblib.load(model_path)
            return True
        except Exception as e:
            st.error(f"Ошибка при загрузке модели для горизонта {horizon}: {str(e)}")
            return False
    
    def load_all_models(self) -> bool:
        """
        Загружает все модели для всех горизонтов.
        
        Returns:
            True если все модели успешно загружены, False иначе
        """
        success_count = 0
        total_count = len(self.model_paths)
        
        for horizon in self.model_paths.keys():
            if self.load_model(horizon):
                success_count += 1
        
        if success_count == total_count:
            st.success(" Все модели LightGBM успешно загружены")
            return True
        elif success_count > 0:
            st.warning(f"️ Загружено {success_count} из {total_count} моделей. Некоторые модели не найдены.")
            st.info("Для работы приложения необходима хотя бы модель для горизонта h=1")
            return success_count > 0
        else:
            st.error(" Не удалось загрузить ни одну модель LightGBM")
            st.info(f"Убедитесь, что модели находятся в: {list(self.model_paths.values())[0].parent}")
            return False
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Загружает метаданные моделей (параметры, признаки).
        
        Returns:
            Словарь с метаданными или None
        """
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                
                # Извлекаем названия признаков
                if "feature_names" in self.metadata:
                    self.feature_names = self.metadata["feature_names"]
                
                return self.metadata
            except Exception as e:
                st.warning(f"Не удалось загрузить метаданные: {str(e)}")
                return None
        return None
    
    def predict(self, features: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Выполняет прогноз на основе признаков для указанного горизонта.
        
        Args:
            features: DataFrame с признаками для прогнозирования
            horizon: Горизонт прогнозирования (1, 24, или 168)
        
        Returns:
            Массив с прогнозами
        """
        if horizon not in self.models:
            raise ValueError(f"Модель для горизонта {horizon} не загружена. Вызовите load_model({horizon}) сначала.")
        
        try:
            model = self.models[horizon]
            
            # Используем feature_names из метаданных, если доступны
            if self.feature_names:
                # Проверяем, что все признаки из метаданных присутствуют
                missing_features = set(self.feature_names) - set(features.columns)
                if missing_features:
                    # Добавляем отсутствующие признаки нулями
                    for col in missing_features:
                        features[col] = 0
                
                # Переупорядочиваем признаки в порядке, ожидаемом моделью
                features = features[self.feature_names]
            
            predictions = model.predict(features)
            # Убеждаемся, что прогнозы неотрицательны (энергопотребление >= 0)
            predictions = np.maximum(predictions, 0)
            return predictions
        except Exception as e:
            st.error(f"Ошибка при прогнозировании для горизонта {horizon}: {str(e)}")
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
        Выполняет прогноз на заданный горизонт используя direct strategy.
        Для LightGBM: использует модель для указанного горизонта, которая предсказывает сразу на h шагов вперед.
        Для многошагового прогноза (например, 168 часов) использует пошаговый подход с моделью h=1.
        
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
        
        # Для пошагового прогноза используем модель h=1
        # Модели для h=24 и h=168 обучены предсказывать значение через 24/168 часов,
        # но для полного прогноза на эти горизонты нужны все промежуточные точки
        if 1 not in self.models:
            raise ValueError("Модель для горизонта h=1 не загружена. Необходима для пошагового прогноза.")
        
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
        
        # Пошаговый прогноз с моделью h=1
        for i, forecast_date in enumerate(forecast_dates):
            # Генерируем признаки для текущего момента
            temp_df = df.copy()
            temp_df = engineer_features(temp_df, date_column, target_column)
            
            # Обновляем time_idx для последовательности (должен быть индексом от начала)
            if "time_idx" in temp_df.columns:
                temp_df["time_idx"] = np.arange(len(temp_df), dtype=np.int32)
            
            # Берём последнюю строку с признаками
            available_features = [col for col in feature_columns if col in temp_df.columns]
            if len(available_features) != len(feature_columns):
                missing = set(feature_columns) - set(available_features)
                if i == 0:  # Предупреждение только один раз
                    st.warning(f"Отсутствуют признаки: {missing}")
                    # Заполняем отсутствующие признаки нулями
                    for col in missing:
                        temp_df[col] = 0
                    available_features = feature_columns.copy()
            
            features = temp_df[available_features].iloc[[-1]]
            
            # Если модель ожидает другой порядок признаков, используем feature_names из метаданных
            if self.feature_names:
                # Проверяем, что все признаки из метаданных присутствуют
                missing_in_features = set(self.feature_names) - set(features.columns)
                if missing_in_features:
                    # Добавляем отсутствующие признаки нулями
                    for col in missing_in_features:
                        features[col] = 0
                
                # Переупорядочиваем признаки в порядке, ожидаемом моделью
                features = features[self.feature_names]
            
            # Выполняем прогноз на 1 шаг вперед
            prediction = self.predict(features, 1)[0]
            
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
        
        # Сортируем по дате и удаляем дубликаты
        result_df = pd.DataFrame(forecasts)
        result_df = result_df.drop_duplicates(subset=[date_column])
        result_df = result_df.sort_values(date_column).reset_index(drop=True)
        
        return result_df
