"""
visualizations.py
------------------
Reusable Plotly chart-building functions. Keeping chart code here avoids
duplicating chart logic across multiple pages (EDA, Performance, etc.)
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Consistent color theme used across every chart in the app
PRIMARY_COLOR = "#4F46E5"
SECONDARY_COLOR = "#F59E0B"
PLACED_COLOR = "#10B981"
NOT_PLACED_COLOR = "#EF4444"


def placement_distribution_pie(df: pd.DataFrame):
    """Pie chart showing overall Placed vs Not Placed ratio."""
    counts = df["Placement"].map({1: "Placed", 0: "Not Placed"}).value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map={"Placed": PLACED_COLOR, "Not Placed": NOT_PLACED_COLOR},
        hole=0.45,
        title="Overall Placement Distribution",
    )
    fig.update_traces(textinfo="percent+label")
    return fig


def numeric_histogram(df: pd.DataFrame, column: str):
    """Histogram of a numeric column, split by placement outcome."""
    plot_df = df.copy()
    plot_df["Placement"] = plot_df["Placement"].map({1: "Placed", 0: "Not Placed"})
    fig = px.histogram(
        plot_df, x=column, color="Placement", barmode="overlay", nbins=30,
        color_discrete_map={"Placed": PLACED_COLOR, "Not Placed": NOT_PLACED_COLOR},
        title=f"Distribution of {column} by Placement Status",
    )
    fig.update_layout(bargap=0.05)
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap across all numeric features + target."""
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(height=550)
    return fig


def scatter_by_placement(df: pd.DataFrame, x: str, y: str):
    """Scatter plot of two numeric features colored by placement outcome."""
    plot_df = df.copy()
    plot_df["Placement"] = plot_df["Placement"].map({1: "Placed", 0: "Not Placed"})
    fig = px.scatter(
        plot_df, x=x, y=y, color="Placement",
        color_discrete_map={"Placed": PLACED_COLOR, "Not Placed": NOT_PLACED_COLOR},
        opacity=0.6,
        title=f"{y} vs {x} (colored by Placement)",
    )
    return fig


def placement_rate_by_category(df: pd.DataFrame, column: str):
    """
    Bar chart of placement RATE (%) grouped by a categorical/binary column
    (e.g. Internship: 0 vs 1).
    """
    grouped = df.groupby(column)["Placement"].mean().reset_index()
    grouped["Placement"] = grouped["Placement"] * 100
    fig = px.bar(
        grouped, x=column, y="Placement",
        text_auto=".1f",
        color="Placement",
        color_continuous_scale=[NOT_PLACED_COLOR, PLACED_COLOR],
        title=f"Placement Rate (%) by {column}",
        labels={"Placement": "Placement Rate (%)"},
    )
    return fig


def box_plot_by_placement(df: pd.DataFrame, column: str):
    """Box plot showing spread of a numeric feature by placement outcome."""
    plot_df = df.copy()
    plot_df["Placement"] = plot_df["Placement"].map({1: "Placed", 0: "Not Placed"})
    fig = px.box(
        plot_df, x="Placement", y=column, color="Placement",
        color_discrete_map={"Placed": PLACED_COLOR, "Not Placed": NOT_PLACED_COLOR},
        title=f"{column} Spread by Placement Status",
    )
    return fig


def confusion_matrix_heatmap(cm: np.ndarray):
    """Heatmap visualization of a 2x2 confusion matrix."""
    labels = ["Not Placed", "Placed"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Predicted: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig.update_layout(title="Confusion Matrix", height=450)
    return fig


def feature_importance_bar(feature_importance_df: pd.DataFrame):
    """Horizontal bar chart of feature importances from the trained model."""
    fig = px.bar(
        feature_importance_df.sort_values("Importance", ascending=True),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis",
        title="Feature Importance (Random Forest)",
    )
    fig.update_layout(height=450)
    return fig


def probability_gauge(probability: float):
    """Gauge chart showing a single student's placement probability (0-100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        title={"text": "Placement Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PRIMARY_COLOR},
            "steps": [
                {"range": [0, 40], "color": "#FEE2E2"},
                {"range": [40, 70], "color": "#FEF3C7"},
                {"range": [70, 100], "color": "#D1FAE5"},
            ],
        },
    ))
    fig.update_layout(height=350)
    return fig
