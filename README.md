# Прогнозирование энергопотребления

Веб-приложение на Streamlit для прогнозирования почасового энергопотребления (кВт·ч)
на основе моделей XGBoost.

## Основные возможности

- Загрузка и валидация CSV данных
- Прогноз на горизонты 1, 24 и 168 часов
- Rolling direct прогноз для построения рядов на 24/168 часов
- Интерактивные графики (Plotly)
- Экспорт результатов в CSV

## Технологический стек

- Python 3.10+
- Streamlit
- XGBoost
- Pandas, NumPy
- Plotly
- Scikit-learn
- Joblib

## Структура проекта

```
EnergyForecast/
├── app.py
├── config/
│   └── settings.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── visualization/
│   └── analysis/
├── artifacts/
│   ├── weights/                  # XGBoost модели
│   │   ├── xgb_final_h1.joblib
│   │   ├── xgb_final_h24.joblib
│   │   └── xgb_final_h168.joblib
│   └── metadata/                 # Опциональные метаданные
├── data/
│   └── samples/
│       └── sample_1000_hours.csv
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_model_tuning.ipynb
│   └── 04_final_training.ipynb
├── ML_analysis.ipynb
├── requirements.txt
└── README.md
```

## Установка

1. Создайте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# или
venv\Scripts\activate     # Windows
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

## Запуск приложения

```bash
streamlit run app.py
```

Откройте в браузере: `http://localhost:8501`

## Использование

1. Нажмите **"Загрузить модели"** в боковой панели.
2. Загрузите CSV на вкладке **"Загрузка данных"**.
3. На вкладке **"Прогнозирование"** выберите горизонт и нажмите **"Выполнить прогноз"**.
4. При необходимости скачайте прогноз в CSV.

## Формат данных

Поддерживаются два варианта входных CSV:

### Формат 1 (рекомендуется)

| datetime | Usage_kWh |
|----------|-----------|
| 2017-01-01 00:00:00 | 1234.56 |

### Формат 2 (автоматически преобразуется)

| Date | Time | Usage_kWh |
|------|------|-----------|
| 2017-01-01 | 00 - 01 | 570.69 |

## Модели

Для каждого горизонта используется отдельная модель XGBoost (direct strategy):

- `artifacts/weights/xgb_final_h1.joblib`
- `artifacts/weights/xgb_final_h24.joblib`
- `artifacts/weights/xgb_final_h168.joblib`

Для горизонтов 24 и 168 часов строится прогнозный ряд через rolling direct,
а также вычисляется точка прогноза на самом горизонте.

## Признаки

Модель использует:

- Временные признаки: `hour`, `day_of_week`, `month`, `day_of_year`, `is_weekend`
- Лаги: `lag_1` ... `lag_24`, `lag_48`, `lag_168`
- Скользящие статистики: `rolling_mean_24`, `rolling_std_24`
- Индекс времени: `time_idx`

## Метрики оценки

- MAE
- RMSE
- sMAPE
- MASE

## Требования

- Python 3.10+
- 4 GB RAM и более
- ~500 MB свободного места

## Лицензия

Дипломный проект.
