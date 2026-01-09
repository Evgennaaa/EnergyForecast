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



