"""
Менеджер данных с тремя уровнями: raw → clean → views.
Избегает копирования данных в памяти, используя ссылки и views.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
import pickle
import hashlib
from functools import lru_cache


class DataManager:
    """
    Менеджер данных с тремя уровнями обработки.
    
    Уровень A (raw): сырые данные - загружаются один раз
    Уровень B (clean): очищенные данные - создаются один раз, кешируются
    Уровень C (views): витрины под модели - создаются по требованию, используют ссылки
    
    Views:
    1. timeseries → Series (для Naive, Seasonal Naive, Holt-Winters, SARIMA)
    2. ml_features → (X, y) (для Random Forest, XGBoost, LightGBM, Linear Regression)
    3. lstm_base → DataFrame (для LSTM - базовые данные без sequences)
    4. prophet → DataFrame (для Prophet - формат ds/y)
    
    Все данные используют оригинальное название колонки: Usage_kWh
    """
    
    def __init__(self, base_dir: Path = None):
        """
        Инициализирует DataManager.
        
        Args:
            base_dir: Базовая директория проекта (по умолчанию определяется автоматически)
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
        
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.clean_dir = self.data_dir / "clean"
        self.views_dir = self.data_dir / "views"
        
        # Создаём директории
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.views_dir.mkdir(parents=True, exist_ok=True)
        
        # Кеш для хранения данных в памяти (избегаем повторной загрузки)
        self._raw_cache: Optional[pd.DataFrame] = None
        self._clean_cache: Optional[pd.DataFrame] = None
        self._views_cache: Dict[str, Any] = {}
    
    # ========== УРОВЕНЬ A: RAW ==========
    
    def load_raw(self, filename: str = "Kharovsklesprom_data.csv", force_reload: bool = False) -> pd.DataFrame:
        """
        Загружает сырые данные (Уровень A).
        Загружает один раз, затем использует кеш.
        
        Args:
            filename: Имя файла в data/raw/ или корне проекта
            force_reload: Принудительная перезагрузка
        
        Returns:
            DataFrame с сырыми данными
        """
        if self._raw_cache is not None and not force_reload:
            return self._raw_cache
        
        # Ищем файл в разных местах
        file_path = self.raw_dir / filename
        if not file_path.exists():
            file_path = self.base_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Файл данных не найден: {filename}\n"
                f"Искали в: {self.raw_dir} и {self.base_dir}"
            )
        
        df = pd.read_csv(file_path, index_col=0)
        self._raw_cache = df
        print(f" Raw данные загружены: {len(df)} строк из {file_path}")
        return df
    
    def get_raw(self) -> pd.DataFrame:
        """Возвращает сырые данные из кеша или загружает."""
        if self._raw_cache is None:
            return self.load_raw()
        return self._raw_cache
    
    # ========== УРОВЕНЬ B: CLEAN ==========
    
    def get_clean(
        self,
        force_recompute: bool = False,
        cleaning_func: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Получает очищенные данные (Уровень B).
        Использует кеш на диске и в памяти.
        
        Args:
            force_recompute: Принудительный пересчёт
            cleaning_func: Функция очистки (если None, используется стандартная)
        
        Returns:
            DataFrame с очищенными данными (индекс DateTime, колонка Usage_kWh)
        """
        cache_file = self.clean_dir / "clean_data.parquet"
        
        # Проверяем кеш в памяти
        if self._clean_cache is not None and not force_recompute:
            return self._clean_cache
        
        # Проверяем кеш на диске
        if cache_file.exists() and not force_recompute:
            try:
                df = pd.read_parquet(cache_file)
                self._clean_cache = df
                print(f" Clean данные загружены из кеша: {len(df)} строк")
                return df
            except Exception as e:
                print(f"️ Ошибка загрузки кеша, пересчитываем: {e}")
        
        # Вычисляем clean данные
        raw_df = self.get_raw()
        
        if cleaning_func is None:
            df = self._default_cleaning(raw_df)
        else:
            df = cleaning_func(raw_df)
        
        # Сохраняем в кеш
        try:
            df.to_parquet(cache_file)
            print(f" Clean данные сохранены в кеш: {cache_file}")
        except Exception as e:
            print(f"️ Не удалось сохранить кеш: {e}")
        
        self._clean_cache = df
        print(f" Clean данные созданы: {len(df)} строк")
        return df
    
    def _default_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Стандартная функция очистки данных.
        Создаёт копию только при необходимости изменения структуры.
        """
        # Преобразуем Date и Time в datetime (это создаёт копию, но необходимо)
        if 'Date' in df.columns and 'Time' in df.columns:
            def parse_time(time_str):
                """Извлекает час из строки времени."""
                if pd.isna(time_str):
                    return "00"
                time_str = str(time_str).strip()
                
                # Формат "00 - 01" или "00-01"
                if " - " in time_str or "-" in time_str:
                    hour = time_str.split("-")[0].strip()
                    return hour.zfill(2)
                
                # Формат "00:00:00" или "00:00"
                if ":" in time_str:
                    return time_str.split(":")[0].zfill(2)
                
                # Если просто число
                try:
                    return str(int(float(time_str))).zfill(2)
                except:
                    return "00"
            
            df = df.copy()  # Копия необходима для изменения структуры
            df['hour_str'] = df['Time'].apply(parse_time)
            df['DateTime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['hour_str'] + ':00:00',
                errors='coerce'
            )
            df.set_index('DateTime', inplace=True)
            df.drop(['Date', 'Time', 'hour_str'], axis=1, inplace=True, errors='ignore')
            df = df.dropna(subset=[df.columns[0]])
        
        # Оставляем оригинальное название колонки Usage_kWh
        # Сортировка (создаёт view, не копию)
        df = df.sort_index()
        
        return df
    
    # ========== УРОВЕНЬ C: VIEWS ==========
    
    def get_timeseries_view(self, force_recompute: bool = False) -> pd.Series:
        """
        View 1: Временной ряд (Series).
        Используется для: Naive, Seasonal Naive, Holt-Winters, SARIMA.
        
        Returns:
            Series с индексом DateTime и значениями Usage_kWh
        """
        cache_key = 'timeseries'
        
        if cache_key in self._views_cache and not force_recompute:
            return self._views_cache[cache_key]
        
        cache_file = self.views_dir / "timeseries.parquet"
        if cache_file.exists() and not force_recompute:
            try:
                df = pd.read_parquet(cache_file)
                series = df['Usage_kWh']
                self._views_cache[cache_key] = series
                print(f" View 'timeseries' загружена из кеша: {len(series)} точек")
                return series
            except Exception as e:
                print(f"️ Ошибка загрузки кеша, пересчитываем: {e}")
        
        clean_df = self.get_clean()
        series = clean_df['Usage_kWh']  # View, не копия
        
        # Сохраняем в кеш (как DataFrame для совместимости с parquet)
        try:
            clean_df[['Usage_kWh']].to_parquet(cache_file)
        except Exception as e:
            print(f"️ Не удалось сохранить кеш: {e}")
        
        self._views_cache[cache_key] = series
        print(f" View 'timeseries' создана: {len(series)} точек")
        return series
    
    def get_ml_features_view(
        self,
        force_recompute: bool = False,
        target_col: str = "Usage_kWh"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        View 2: ML признаки (DataFrame) + отдельно y/targets.
        Используется для: Random Forest, XGBoost, LightGBM, Linear Regression.
        
        Args:
            force_recompute: Принудительный пересчёт
            target_col: Название колонки с целевой переменной
        
        Returns:
            Tuple (X, y) где X - DataFrame с признаками, y - Series
        """
        from src.data.feature_builder import build_features_X_optimized
        
        cache_key = 'ml_features'
        
        if cache_key in self._views_cache and not force_recompute:
            return self._views_cache[cache_key]
        
        cache_file_X = self.views_dir / "ml_features_X.parquet"
        cache_file_y = self.views_dir / "ml_features_y.parquet"
        
        if cache_file_X.exists() and cache_file_y.exists() and not force_recompute:
            try:
                X = pd.read_parquet(cache_file_X)
                y = pd.read_parquet(cache_file_y)['Usage_kWh']
                result = (X, y)
                self._views_cache[cache_key] = result
                print(f" View 'ml_features' загружена из кеша: X.shape={X.shape}")
                return result
            except Exception as e:
                print(f"️ Ошибка загрузки кеша, пересчитываем: {e}")
        
        clean_df = self.get_clean()
        X, y = build_features_X_optimized(clean_df, target_col=target_col)
        
        # Сохраняем в кеш
        try:
            X.to_parquet(cache_file_X)
            pd.DataFrame({'Usage_kWh': y}).to_parquet(cache_file_y)
        except Exception as e:
            print(f"️ Не удалось сохранить кеш: {e}")
        
        result = (X, y)
        self._views_cache[cache_key] = result
        print(f" View 'ml_features' создана: X.shape={X.shape}")
        return result
    
    def get_lstm_base_view(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        View 3: LSTM базовые данные (DataFrame с одной колонкой).
        Используется для: LSTM (после этого создаются sequences в коде обучения).
        
        Returns:
            DataFrame с колонкой Usage_kWh и индексом DateTime
        """
        cache_key = 'lstm_base'
        
        if cache_key in self._views_cache and not force_recompute:
            return self._views_cache[cache_key]
        
        cache_file = self.views_dir / "lstm_base.parquet"
        if cache_file.exists() and not force_recompute:
            try:
                df = pd.read_parquet(cache_file)
                self._views_cache[cache_key] = df
                print(f" View 'lstm_base' загружена из кеша: {len(df)} строк")
                return df
            except Exception as e:
                print(f"️ Ошибка загрузки кеша, пересчитываем: {e}")
        
        clean_df = self.get_clean()
        df = clean_df[['Usage_kWh']]  # View, не копия
        
        # Сохраняем в кеш
        try:
            df.to_parquet(cache_file)
        except Exception as e:
            print(f"️ Не удалось сохранить кеш: {e}")
        
        self._views_cache[cache_key] = df
        print(f" View 'lstm_base' создана: {len(df)} строк")
        return df
    
    def get_prophet_view(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        View 4: Prophet формат (DataFrame с колонками ds и y).
        Используется для: Prophet.
        
        Returns:
            DataFrame с колонками 'ds' (datetime) и 'y' (target)
        """
        cache_key = 'prophet'
        
        if cache_key in self._views_cache and not force_recompute:
            return self._views_cache[cache_key]
        
        cache_file = self.views_dir / "prophet.parquet"
        if cache_file.exists() and not force_recompute:
            try:
                df = pd.read_parquet(cache_file)
                self._views_cache[cache_key] = df
                print(f" View 'prophet' загружена из кеша: {len(df)} строк")
                return df
            except Exception as e:
                print(f"️ Ошибка загрузки кеша, пересчитываем: {e}")
        
        clean_df = self.get_clean()
        # Копия необходима для reset_index и rename
        result = clean_df.reset_index().rename(columns={clean_df.index.name or "index": "ds", "Usage_kWh": "y"})
        
        # Сохраняем в кеш
        try:
            result.to_parquet(cache_file)
        except Exception as e:
            print(f"️ Не удалось сохранить кеш: {e}")
        
        self._views_cache[cache_key] = result
        print(f" View 'prophet' создана: {len(result)} строк")
        return result
    
    # ========== УТИЛИТЫ ==========
    
    def clear_cache(self, level: Optional[str] = None):
        """
        Очищает кеш.
        
        Args:
            level: 'raw', 'clean', 'views' или None (все)
        """
        if level is None or level == 'raw':
            self._raw_cache = None
        if level is None or level == 'clean':
            self._clean_cache = None
        if level is None or level == 'views':
            self._views_cache.clear()
        print(f" Кеш очищен: {level or 'все уровни'}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Возвращает информацию об использовании памяти."""
        usage = {
            'raw': self._raw_cache.memory_usage(deep=True).sum() if self._raw_cache is not None else 0,
            'clean': self._clean_cache.memory_usage(deep=True).sum() if self._clean_cache is not None else 0,
            'views': {}
        }
        
        for name, view in self._views_cache.items():
            if isinstance(view, tuple):
                # Для ml_features (X, y)
                view_size = sum(v.memory_usage(deep=True).sum() if hasattr(v, 'memory_usage') else 0 for v in view)
            elif hasattr(view, 'memory_usage'):
                view_size = view.memory_usage(deep=True).sum()
            else:
                view_size = 0
            usage['views'][name] = view_size
        
        usage['total'] = usage['raw'] + usage['clean'] + sum(usage['views'].values())
        return usage

