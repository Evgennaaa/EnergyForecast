"""
Главный файл Streamlit приложения для прогнозирования энергопотребления.
Shadcn-ui inspired design.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from config.settings import (
    SAMPLE_DATA_PATH, XGB_MODEL_PATHS,
    FORECAST_HORIZONS, DATE_COLUMN, TARGET_COLUMN,
    FEATURE_COLUMNS, PAGE_TITLE, PAGE_ICON
)
from src.data.loader import load_csv, load_uploaded_file, ensure_datetime_column, normalize_data_format
from src.data.validator import validate_dataframe, validate_forecast_horizon, check_data_quality
from src.data.preprocessor import prepare_data_for_forecast
from src.models.xgb_manager import XGBModelManager
from src.evaluation.metrics import calculate_all_metrics, format_metrics
from src.visualization.plots import (
    plot_historical_data, plot_forecast, plot_metrics_comparison,
    plot_daily_profile, plot_weekly_profile, plot_distribution
)
from src.analysis.data_analysis import (
    build_daily_profile, build_weekly_profile,
    calculate_basic_statistics, detect_outliers_iqr, get_data_period
)


# Настройка страницы
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Дипломный проект: Прогнозирование энергопотребления"
    }
)

# Загрузка кастомных стилей
def load_custom_css():
    """Загружает кастомные CSS стили в стиле shadcn-ui"""
    css_path = Path(__file__).parent / ".streamlit" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# Заголовок приложения с улучшенным дизайном
st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>Прогнозирование энергопотребления</h1>
        <p style='color: rgb(148, 163, 184); font-size: 1rem; margin-top: 0.5rem;'>
            Прогнозирование почасового энергопотребления с применением технологий машинного обучения
        </p>
    </div>
""", unsafe_allow_html=True)

# Инициализация состояния сессии
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_manager" not in st.session_state:
    st.session_state.model_manager = None
if "point_forecast_df" not in st.session_state:
    st.session_state.point_forecast_df = None
if "missing_ratio_too_high" not in st.session_state:
    st.session_state.missing_ratio_too_high = False
if "historical_data" not in st.session_state:
    st.session_state.historical_data = None


# Вспомогательные функции для UI компонентов
def render_card(title: str, content, icon: str = None):
    """Создаёт карточку в стиле shadcn-ui"""
    icon_html = f"<span style='font-size: 1.5rem; margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    st.markdown(f"""
    <div style='
        background: rgb(30, 41, 59);
        border: 1px solid rgb(51, 65, 85);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    '>
        <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
            {icon_html}
            <h3 style='margin: 0; color: rgb(241, 245, 249); font-weight: 600;'>{title}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return content

def render_badge(text: str, variant: str = "default"):
    """Создаёт бейдж в стиле shadcn-ui"""
    colors = {
        "default": "rgb(51, 65, 85)",
        "primary": "rgb(14, 165, 233)",
        "success": "rgb(34, 197, 94)",
        "warning": "rgb(234, 179, 8)",
        "error": "rgb(239, 68, 68)"
    }
    return st.markdown(f"""
    <span style='
        background: {colors.get(variant, colors["default"])};
        color: rgb(255, 255, 255);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    '>{text}</span>
    """, unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h2 style='margin-bottom: 0.5rem; color: rgb(241, 245, 249);'>Настройки</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Загрузка модели
    st.markdown("### Загрузка модели")
    st.markdown("**XGBoost**")

    if st.button("Загрузить модели", type="primary", use_container_width=True):
        xgb_manager = XGBModelManager(XGB_MODEL_PATHS)
        if xgb_manager.load_all_models():
            st.session_state.model_manager = xgb_manager
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False
    
    # Статус модели
    if st.session_state.model_loaded:
        model_name = "XGBoost"
        st.markdown(f"""
        <div style='
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin-top: 1rem;
            color: rgb(74, 222, 128);
        '>
            Модель <strong>{model_name}</strong> загружена
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
            background: rgba(234, 179, 8, 0.1);
            border: 1px solid rgba(234, 179, 8, 0.3);
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin-top: 1rem;
            color: rgb(253, 224, 71);
        '>
            Модель не загружена
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgb(51, 65, 85); margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Информация о проекте
    st.markdown("### О проекте")
    st.markdown("""
    <div style='
        background: rgb(30, 41, 59);
        border: 1px solid rgb(51, 65, 85);
        border-radius: 0.5rem;
        padding: 1rem;
        font-size: 0.875rem;
        line-height: 1.75;
        color: rgb(148, 163, 184);
    '>
        <p style='margin: 0 0 0.75rem 0;'><strong style='color: rgb(241, 245, 249);'>Дипломный проект</strong></p>
        <p style='margin: 0 0 0.75rem 0;'>Прогнозирование почасового энергопотребления</p>
        <div style='margin-top: 1rem;'>
            <p style='margin: 0.5rem 0;'><strong style='color: rgb(241, 245, 249);'>Модель:</strong> XGBoost</p>
            <p style='margin: 0.5rem 0;'><strong style='color: rgb(241, 245, 249);'>Горизонты:</strong> 1ч, 24ч, 168ч</p>
            <p style='margin: 0.5rem 0;'><strong style='color: rgb(241, 245, 249);'>Стратегия:</strong> Прямая</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Основной контент
tab1, tab2, tab3, tab4 = st.tabs([
    "Загрузка данных", 
    "Анализ данных", 
    "Прогнозирование", 
    "Визуализация"
])

# Вкладка 1: Загрузка данных
with tab1:
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h2 style='margin-bottom: 0.5rem; color: rgb(241, 245, 249);'>Загрузка и валидация данных</h2>
        <p style='color: rgb(148, 163, 184); font-size: 0.875rem;'>
            Загрузите CSV файл с данными энергопотребления или используйте пример данных
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Выбор источника данных
    data_source = st.radio(
        "Выберите источник данных:",
        ["Загрузить файл", "Использовать пример данных"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Выберите CSV файл",
            type=["csv"],
            help="Поддерживаемые форматы:\n1) datetime, Usage_kWh\n2) Date, Time, Usage_kWh"
        )
        
        if uploaded_file is not None:
            try:
                df = load_uploaded_file(uploaded_file)
                st.success("Файл успешно загружен")
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {str(e)}")
    
    else:  # Использовать пример данных
        if SAMPLE_DATA_PATH.exists():
            try:
                df = load_csv(SAMPLE_DATA_PATH)
                st.success("Пример данных загружен")
            except Exception as e:
                st.error(f"Ошибка при загрузке примера: {str(e)}")
        else:
            st.warning("Файл с примером данных не найден. Пожалуйста, загрузите свой файл.")
    
    # Обработка загруженных данных
    if df is not None:
        # Показываем информацию о формате данных
        original_columns = list(df.columns)
        st.info(f"Обнаружены колонки: {', '.join(original_columns)}")
        
        # Нормализация формата данных (поддержка Date/Time/Usage_kWh и datetime/Usage_kWh)
        df = normalize_data_format(df)
        
        # Показываем результат преобразования
        if "Date" in original_columns or "Time" in original_columns or "Usage_kWh" in original_columns:
            st.success("Данные преобразованы в стандартный формат (datetime, Usage_kWh)")
        
        # Преобразование даты
        df = ensure_datetime_column(df, DATE_COLUMN)
        
        # Валидация
        is_valid, error_msg = validate_dataframe(
            df, [DATE_COLUMN], DATE_COLUMN, TARGET_COLUMN
        )
        
        if is_valid:
            # Проверяем пропуски до обработки
            missing_before = df[TARGET_COLUMN].isna().sum() if TARGET_COLUMN in df.columns else 0
            missing_ratio = (missing_before / len(df)) if len(df) > 0 else 0
            st.session_state.missing_ratio_too_high = missing_before > 0 and missing_ratio > 0.02

            # Предобработка
            if st.session_state.missing_ratio_too_high:
                df = prepare_data_for_forecast(df, DATE_COLUMN, None)
            else:
                df = prepare_data_for_forecast(df, DATE_COLUMN, TARGET_COLUMN)
            
            # Сохранение в сессию
            st.session_state.historical_data = df
            
            # Статистика данных
            st.subheader("Статистика данных")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Количество записей", len(df))
            with col2:
                st.metric("Период", f"{df[DATE_COLUMN].min().date()} - {df[DATE_COLUMN].max().date()}")
            with col3:
                if TARGET_COLUMN in df.columns:
                    st.metric("Среднее (кВт·ч)", f"{df[TARGET_COLUMN].mean():.2f}")
            with col4:
                if TARGET_COLUMN in df.columns:
                    st.metric("Максимум (кВт·ч)", f"{df[TARGET_COLUMN].max():.2f}")
            
            # Качество данных
            quality_stats = check_data_quality(df, TARGET_COLUMN)
            missing_after = quality_stats["missing_values"]
            if st.session_state.missing_ratio_too_high:
                st.warning(
                    "В данных много пропусков. "
                    f"Доля пропусков: {missing_ratio:.2%} (всего {missing_before}). "
                    "Автоматическое заполнение не выполнено. Прогнозирование будет заблокировано."
                )
            elif missing_before > 0:
                st.info(
                    "Обнаружены пропуски в данных. "
                    f"До обработки: {missing_before}, после обработки: {missing_after}. "
                    "Применено заполнение пропусков последним значением."
                )
            elif missing_after > 0:
                st.warning(f"Обнаружено {missing_after} пропущенных значений после обработки")
            
            # Превью данных
            st.subheader("Превью данных")
            st.dataframe(df.head(100), use_container_width=True)
            
            # График исторических данных
            if TARGET_COLUMN in df.columns:
                st.subheader("График исторических данных")
                fig_hist = plot_historical_data(df, DATE_COLUMN, TARGET_COLUMN)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        else:
            st.error(f"Ошибка валидации: {error_msg}")


# Вкладка 2: Анализ данных
with tab2:
    st.header("Анализ данных энергопотребления")
    
    if st.session_state.historical_data is None:
        st.warning("️ Пожалуйста, загрузите данные на вкладке 'Загрузка данных'.")
    else:
        df = st.session_state.historical_data.copy()
        
        # Базовая статистика
        st.subheader("Базовая статистика")
        stats = calculate_basic_statistics(df, TARGET_COLUMN)
        period_info = get_data_period(df, DATE_COLUMN)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Среднее", f"{stats['mean']:.2f} кВт·ч")
        with col2:
            st.metric("Медиана", f"{stats['median']:.2f} кВт·ч")
        with col3:
            st.metric("Стандартное отклонение", f"{stats['std']:.2f} кВт·ч")
        with col4:
            st.metric("IQR", f"{stats['iqr']:.2f} кВт·ч")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Минимум", f"{stats['min']:.2f} кВт·ч")
        with col2:
            st.metric("Максимум", f"{stats['max']:.2f} кВт·ч")
        with col3:
            st.metric("25-й перцентиль", f"{stats['q25']:.2f} кВт·ч")
        with col4:
            st.metric("75-й перцентиль", f"{stats['q75']:.2f} кВт·ч")
        
        # Информация о периоде
        st.subheader("Информация о периоде данных")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Начало:** {period_info['start']}")
        with col2:
            st.info(f"**Конец:** {period_info['end']}")
        with col3:
            st.info(f"**Дней:** {period_info['days']}")
        
        # Распределение значений
        st.subheader("Распределение значений")
        fig_dist = plot_distribution(df, TARGET_COLUMN)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Выбросы
        st.subheader("Анализ выбросов")
        outliers = detect_outliers_iqr(df, TARGET_COLUMN, factor=2.0)
        outlier_percentage = (len(outliers) / len(df)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Количество выбросов", len(outliers))
        with col2:
            st.metric("Процент выбросов", f"{outlier_percentage:.2f}%")
        
        if len(outliers) > 0:
            st.dataframe(outliers[[DATE_COLUMN, TARGET_COLUMN]].head(20), use_container_width=True)
            st.caption("Выбросы представляют собой реальные пиковые нагрузки и плановые остановки, а не ошибки измерения.")
        else:
            st.success("Выбросов не обнаружено (метод IQR с коэффициентом 2.0)")
        
        # Суточный профиль
        st.subheader("Суточный профиль энергопотребления")
        try:
            workday_profile, weekend_profile = build_daily_profile(df, DATE_COLUMN, TARGET_COLUMN)
            fig_daily = plot_daily_profile(workday_profile, weekend_profile)
            st.plotly_chart(fig_daily, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка при построении суточного профиля: {str(e)}")
        
        # Недельный профиль
        st.subheader("Недельный профиль энергопотребления")
        try:
            df_week = ensure_datetime_column(df.copy(), DATE_COLUMN)
            df_week = df_week.sort_values(DATE_COLUMN)

            if df_week.empty:
                st.warning("Нет данных для построения недельного профиля.")
            else:
                last_date = df_week[DATE_COLUMN].max()
                start_date = last_date - pd.Timedelta(days=7)
                last_week_df = df_week[df_week[DATE_COLUMN] >= start_date]

                if last_week_df.empty:
                    st.warning("Недостаточно данных для последних 7 дней.")
                else:
                    fig_weekly = plot_historical_data(
                        last_week_df,
                        DATE_COLUMN,
                        TARGET_COLUMN,
                        title="Потребление за последние 7 дней",
                        height=400
                    )
                    st.plotly_chart(fig_weekly, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка при построении недельного профиля: {str(e)}")


# Вкладка 3: Прогнозирование
with tab3:
    st.header("Прогнозирование энергопотребления")
    
    if not st.session_state.model_loaded:
        st.warning("Пожалуйста, загрузите модель в боковой панели.")
    elif st.session_state.missing_ratio_too_high:
        st.warning("Прогнозирование заблокировано: доля пропусков в данных превышает 2%.")
    elif st.session_state.historical_data is None:
        st.warning("Пожалуйста, загрузите данные на вкладке 'Загрузка данных'.")
    else:
        df = st.session_state.historical_data.copy()
        
        # Выбор горизонта прогнозирования
        st.subheader("Параметры прогнозирования")
        horizon_name = st.selectbox(
            "Выберите горизонт прогнозирования:",
            options=list(FORECAST_HORIZONS.keys()),
            index=1  # По умолчанию 24 часа
        )
        horizon = FORECAST_HORIZONS[horizon_name]
        
        # Валидация горизонта
        is_valid_horizon, horizon_error = validate_forecast_horizon(horizon)
        if not is_valid_horizon:
            st.error(f"{horizon_error}")
        
        # Кнопка прогнозирования
        if st.button("Выполнить прогноз", type="primary"):
            with st.spinner("Выполняется прогнозирование..."):
                try:
                    # Используем XGBoost модели для выбранного горизонта
                    manager = st.session_state.model_manager
                    if horizon not in manager.models:
                        if not manager.load_model(horizon):
                            st.error(f"Не удалось загрузить модель для горизонта {horizon}")
                            st.stop()

                    # Для рядов на 24/168 часов используем rolling direct прогноз (модель горизонта)
                    if horizon > 1:
                        point_forecast_df = manager.predict_forecast(
                            historical_data=df,
                            horizon=horizon,
                            feature_columns=FEATURE_COLUMNS,
                            date_column=DATE_COLUMN,
                            target_column=TARGET_COLUMN
                        )
                        forecast_df = manager.predict_rolling_direct_series(
                            historical_data=df,
                            horizon=horizon,
                            n_points=horizon,
                            feature_columns=FEATURE_COLUMNS,
                            date_column=DATE_COLUMN,
                            target_column=TARGET_COLUMN
                        )
                        st.session_state.point_forecast_df = point_forecast_df
                    else:
                        forecast_df = manager.predict_forecast(
                            historical_data=df,
                            horizon=horizon,
                            feature_columns=FEATURE_COLUMNS,
                            date_column=DATE_COLUMN,
                            target_column=TARGET_COLUMN
                        )
                        st.session_state.point_forecast_df = None
                    
                    # Сохранение прогноза в сессию
                    st.session_state.forecast_df = forecast_df
                    st.session_state.forecast_horizon = horizon
                    
                    st.success(f"Прогноз выполнен успешно для горизонта {horizon_name}")
                    
                    # Статистика прогноза
                    st.subheader("Статистика прогноза")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Средний прогноз (кВт·ч)", f"{forecast_df['forecast'].mean():.2f}")
                    with col2:
                        st.metric("Минимальный прогноз (кВт·ч)", f"{forecast_df['forecast'].min():.2f}")
                    with col3:
                        st.metric("Максимальный прогноз (кВт·ч)", f"{forecast_df['forecast'].max():.2f}")
                    
                    # Таблица с прогнозом
                    if horizon > 1:
                        st.subheader("Таблица прогноза (ряд)")
                        st.dataframe(forecast_df, use_container_width=True)

                        if st.session_state.point_forecast_df is not None:
                            st.subheader("Точка на горизонте прогноза")
                            st.dataframe(st.session_state.point_forecast_df, use_container_width=True)
                    else:
                        st.subheader("Таблица прогноза")
                        st.dataframe(forecast_df, use_container_width=True)
                    
                    # Кнопка скачивания
                    csv = forecast_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать прогноз (CSV)",
                        data=csv,
                        file_name=f"forecast_{horizon}h.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {str(e)}")
                    st.exception(e)
        
        # Отображение сохранённого прогноза
        if "forecast_df" in st.session_state:
            st.subheader("Последний прогноз")
            forecast_df = st.session_state.forecast_df
            display_horizon = st.session_state.get("forecast_horizon")
            horizon_labels = {v: k for k, v in FORECAST_HORIZONS.items()}
            display_label = horizon_labels.get(display_horizon, f"{display_horizon} ч")

            if display_horizon and display_horizon > 1:
                history_tail = st.session_state.historical_data.tail(1)
                fig_forecast = plot_forecast(
                    history_tail, forecast_df, DATE_COLUMN, TARGET_COLUMN,
                    title=f"Прогноз энергопотребления ({display_label})"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)


# Вкладка 4: Визуализация
with tab4:
    st.header("Визуализация результатов")
    
    if st.session_state.historical_data is None:
        st.warning("Пожалуйста, загрузите данные на вкладке 'Загрузка данных'.")
    elif "forecast_df" not in st.session_state:
        st.warning("Пожалуйста, выполните прогноз на вкладке 'Прогнозирование'.")
    else:
        df = st.session_state.historical_data
        forecast_df = st.session_state.forecast_df
        
        # График прогноза
        st.subheader("График прогноза")
        fig_forecast = plot_forecast(
            df, forecast_df, DATE_COLUMN, TARGET_COLUMN,
            title="Прогноз энергопотребления"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Дополнительные визуализации
        st.subheader("Дополнительная информация")
        
        # Сравнение последних исторических значений с началом прогноза
        if len(df) > 0 and len(forecast_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Последнее историческое значение",
                    f"{df[TARGET_COLUMN].iloc[-1]:.2f} кВт·ч"
                )
            
            with col2:
                st.metric(
                    "Первый прогноз",
                    f"{forecast_df['forecast'].iloc[0]:.2f} кВт·ч"
                )
            
            # Разница
            diff = forecast_df['forecast'].iloc[0] - df[TARGET_COLUMN].iloc[-1]
            st.metric(
                "Разница (первый прогноз - последнее значение)",
                f"{diff:.2f} кВт·ч",
                delta=f"{diff:.2f} кВт·ч"
            )


# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Дипломный проект | Прогнозирование энергопотребления</div>",
    unsafe_allow_html=True
)
