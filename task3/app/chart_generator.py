import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def generate_charts(df: pd.DataFrame, max_rows: int = 10_000) -> list[str]:
    """Генерирует HTML-фрагменты Plotly-графиков."""
    charts = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    numeric_cols = [c for c in numeric_cols if c.lower() not in ("id", "index", "unnamed: 0")]

    plot_df = df.sample(min(max_rows, len(df)), random_state=42) if len(df) > max_rows else df

    if numeric_cols:
        sorted_by_std = sorted(numeric_cols, key=lambda c: df[c].std(), reverse=True)
        for col in sorted_by_std[:6]:
            fig = px.histogram(
                plot_df, x=col,
                title=f"Распределение: {col}",
                template="plotly_white",
                color_discrete_sequence=["#636EFA"],
            )
            fig.update_layout(
                margin=dict(l=40, r=20, t=40, b=40),
                height=350,
                xaxis_title=col,
                yaxis_title="Количество",
            )
            charts.append({
                "title": f"Распределение: {col}",
                "html": pio.to_html(fig, full_html=False, include_plotlyjs="cdn"),
                "type": "histogram",
            })

    if len(numeric_cols) > 2:
        corr_cols = numeric_cols[:15]
        corr = df[corr_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
        ))
        fig.update_layout(
            title="Матрица корреляций",
            template="plotly_white",
            margin=dict(l=40, r=20, t=40, b=40),
            height=500,
        )
        charts.append({
            "title": "Матрица корреляций",
            "html": pio.to_html(fig, full_html=False, include_plotlyjs=False),
            "type": "heatmap",
        })

    for col in categorical_cols[:4]:
        counts = df[col].value_counts().head(10)
        fig = px.bar(
            x=counts.index.astype(str), y=counts.values,
            title=f"Распределение: {col}",
            template="plotly_white",
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
            height=350,
            xaxis_title=col,
            yaxis_title="Количество",
        )
        charts.append({
            "title": f"Распределение: {col}",
            "html": pio.to_html(fig, full_html=False, include_plotlyjs=False),
            "type": "bar",
        })

    if len(numeric_cols) >= 2:
        for col in numeric_cols[:4]:
            fig = px.box(
                plot_df, y=col,
                title=f"Выбросы: {col}",
                template="plotly_white",
                color_discrete_sequence=["#00CC96"],
            )
            fig.update_layout(
                margin=dict(l=40, r=20, t=40, b=40),
                height=300,
            )
            charts.append({
                "title": f"Выбросы: {col}",
                "html": pio.to_html(fig, full_html=False, include_plotlyjs=False),
                "type": "box",
            })

    return charts
