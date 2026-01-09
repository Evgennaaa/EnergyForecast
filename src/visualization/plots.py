"""
Модуль для визуализации данных и прогнозов с использованием Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


def plot_historical_data(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    title: str = "Исторические данные энергопотребления",
    height: int = 500
) -> go.Figure:
    """
    Строит график исторических данных энергопотребления.
    
    Args:
        df: DataFrame с историческими данными
        date_column: Название колонки с датой
        target_column: Название колонки с целевой переменной
        title: Заголовок графика
        height: Высота графика в пикселях
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df[target_column],
        mode="lines",
        name="Энергопотребление",
        line=dict(color="blue", width=1.5),
        hovertemplate="<b>Дата:</b> %{x}<br><b>Энергопотребление:</b> %{y:.2f} кВт·ч<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Дата и время",
        yaxis_title="Энергопотребление (кВт·ч)",
        height=height,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_column: str,
    target_column: str,
    forecast_column: str = "forecast",
    title: str = "Прогноз энергопотребления",
    height: int = 500
) -> go.Figure:
    """
    Строит график прогноза вместе с историческими данными.
    
    Args:
        historical_df: DataFrame с историческими данными
        forecast_df: DataFrame с прогнозами
        date_column: Название колонки с датой
        target_column: Название колонки с целевой переменной (исторические данные)
        forecast_column: Название колонки с прогнозом
        title: Заголовок графика
        height: Высота графика в пикселях
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Исторические данные
    fig.add_trace(go.Scatter(
        x=historical_df[date_column],
        y=historical_df[target_column],
        mode="lines",
        name="Исторические данные",
        line=dict(color="blue", width=1.5),
        hovertemplate="<b>Дата:</b> %{x}<br><b>Энергопотребление:</b> %{y:.2f} кВт·ч<extra></extra>"
    ))
    
    # Прогноз
    fig.add_trace(go.Scatter(
        x=forecast_df[date_column],
        y=forecast_df[forecast_column],
        mode="lines",
        name="Прогноз",
        line=dict(color="red", width=2, dash="dash"),
        hovertemplate="<b>Дата:</b> %{x}<br><b>Прогноз:</b> %{y:.2f} кВт·ч<extra></extra>"
    ))
    
    # Вертикальная линия, разделяющая историю и прогноз
    if len(historical_df) > 0:
        last_historical_date = historical_df[date_column].iloc[-1]
        
        # Используем add_shape вместо add_vline для совместимости с новыми версиями pandas
        # add_shape работает напрямую с Timestamp объектами
        fig.add_shape(
            type="line",
            x0=last_historical_date,
            x1=last_historical_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dot")
        )
        
        # Добавляем аннотацию отдельно
        fig.add_annotation(
            x=last_historical_date,
            y=1,
            yref="paper",
            text="Начало прогноза",
            showarrow=False,
            xanchor="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Дата и время",
        yaxis_title="Энергопотребление (кВт·ч)",
        height=height,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def plot_forecast_with_confidence(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_column: str,
    target_column: str,
    forecast_column: str = "forecast",
    lower_bound: Optional[pd.Series] = None,
    upper_bound: Optional[pd.Series] = None,
    title: str = "Прогноз энергопотребления с доверительным интервалом",
    height: int = 500
) -> go.Figure:
    """
    Строит график прогноза с доверительным интервалом.
    
    Args:
        historical_df: DataFrame с историческими данными
        forecast_df: DataFrame с прогнозами
        date_column: Название колонки с датой
        target_column: Название колонки с целевой переменной
        forecast_column: Название колонки с прогнозом
        lower_bound: Нижняя граница доверительного интервала
        upper_bound: Верхняя граница доверительного интервала
        title: Заголовок графика
        height: Высота графика в пикселях
    
    Returns:
        Plotly Figure
    """
    fig = plot_forecast(
        historical_df, forecast_df, date_column, target_column,
        forecast_column, title, height
    )
    
    # Добавляем доверительный интервал, если он предоставлен
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df[date_column],
            y=upper_bound,
            mode="lines",
            name="Верхняя граница",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df[date_column],
            y=lower_bound,
            mode="lines",
            name="Доверительный интервал",
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.1)",
            line=dict(width=0),
            hovertemplate="<b>Дата:</b> %{x}<br><b>Интервал:</b> [%{y:.2f}, %{customdata:.2f}] кВт·ч<extra></extra>",
            customdata=upper_bound
        ))
    
    return fig


def plot_metrics_comparison(metrics_dict: dict, title: str = "Сравнение метрик") -> go.Figure:
    """
    Строит столбчатую диаграмму метрик.
    
    Args:
        metrics_dict: Словарь с метриками
        title: Заголовок графика
    
    Returns:
        Plotly Figure
    """
    # Фильтруем метрики, исключая MASE (если он NaN) и нормализуем для визуализации
    plot_metrics = {}
    for key, value in metrics_dict.items():
        if key == "MASE" and (np.isnan(value) or value == 0):
            continue
        if key == "sMAPE":
            plot_metrics[key] = value  # sMAPE уже в процентах
        else:
            plot_metrics[key] = value
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(plot_metrics.keys()),
        y=list(plot_metrics.values()),
        marker_color="steelblue",
        text=[f"{v:.2f}" for v in plot_metrics.values()],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Значение: %{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Метрика",
        yaxis_title="Значение",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_daily_profile(
    workday_profile: pd.Series,
    weekend_profile: pd.Series,
    title: str = "Суточный профиль энергопотребления",
    height: int = 500
) -> go.Figure:
    """
    Строит график суточного профиля для рабочих дней и выходных.
    
    Args:
        workday_profile: Профиль для рабочих дней (Series с индексом = час)
        weekend_profile: Профиль для выходных (Series с индексом = час)
        title: Заголовок графика
        height: Высота графика в пикселях
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=workday_profile.index,
        y=workday_profile.values,
        mode="lines+markers",
        name="Рабочий день",
        line=dict(color="blue", width=2),
        marker=dict(size=6),
        hovertemplate="<b>Час:</b> %{x}<br><b>Потребление:</b> %{y:.2f} кВт·ч<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=weekend_profile.index,
        y=weekend_profile.values,
        mode="lines+markers",
        name="Выходной день",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=6, symbol="square"),
        hovertemplate="<b>Час:</b> %{x}<br><b>Потребление:</b> %{y:.2f} кВт·ч<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Час суток",
        yaxis_title="Среднее потребление (кВт·ч)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0, 23]),
        height=height,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def plot_weekly_profile(
    weekly_profile: pd.Series,
    title: str = "Средний уровень энергопотребления по дням недели",
    height: int = 400
) -> go.Figure:
    """
    Строит график недельного профиля.
    
    Args:
        weekly_profile: Профиль по дням недели (Series с индексом 0-6)
        title: Заголовок графика
        height: Высота графика в пикселях
    
    Returns:
        Plotly Figure
    """
    days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    day_labels = [days[i] for i in weekly_profile.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=day_labels,
        y=weekly_profile.values,
        mode="lines+markers",
        line=dict(color="steelblue", width=2),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Потребление: %{y:.2f} кВт·ч<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="День недели",
        yaxis_title="Среднее потребление (кВт·ч)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False
    )
    
    return fig


def plot_distribution(
    df: pd.DataFrame,
    target_column: str,
    title: str = "Распределение значений энергопотребления",
    height: int = 400
) -> go.Figure:
    """
    Строит гистограмму распределения значений.
    
    Args:
        df: DataFrame с данными
        target_column: Название колонки с целевой переменной
        title: Заголовок графика
        height: Высота графика в пикселях
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[target_column],
        nbinsx=50,
        marker_color="steelblue",
        hovertemplate="<b>Потребление:</b> %{x:.2f} кВт·ч<br><b>Частота:</b> %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Энергопотребление (кВт·ч)",
        yaxis_title="Частота",
        height=height,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Анализ остатков"
) -> go.Figure:
    """
    Строит график остатков (разница между истинными и прогнозируемыми значениями).
    
    Args:
        y_true: Истинные значения
        y_pred: Прогнозируемые значения
        title: Заголовок графика
    
    Returns:
        Plotly Figure с двумя подграфиками
    """
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Остатки по времени", "Распределение остатков"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # График остатков по времени
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(residuals)),
            y=residuals,
            mode="markers",
            name="Остатки",
            marker=dict(color="blue", size=4),
            hovertemplate="<b>Индекс:</b> %{x}<br><b>Остаток:</b> %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Линия нуля
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Гистограмма остатков
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name="Распределение",
            marker_color="steelblue",
            hovertemplate="<b>Остаток:</b> %{x:.2f}<br><b>Частота:</b> %{y}<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Индекс", row=1, col=1)
    fig.update_yaxes(title_text="Остаток (кВт·ч)", row=1, col=1)
    fig.update_xaxes(title_text="Остаток (кВт·ч)", row=1, col=2)
    fig.update_yaxes(title_text="Частота", row=1, col=2)
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig
