import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOT_TYPES


def plot_data_visualization(data: pd.DataFrame, target: str) -> None:
    """
    Plot data visualizations based on user selection.

    Args:
        data (pd.DataFrame): The dataset to visualize.
        target (str): The target column name.
    """
    try:
        st.subheader("Data Visualization")

        # Select visualization type
        viz_type = st.selectbox("Select Visualization Type", PLOT_TYPES)

        if viz_type == "Histogram":
            x_feature = st.selectbox("Select the feature for Histogram:", data.columns)
            fig = px.histogram(data, x=x_feature, title=f"Histogram of {x_feature}")
            st.plotly_chart(fig)

        elif viz_type == "Scatter Plot":
            x_feature = st.selectbox("Select the X-axis feature:", data.columns)
            y_feature = st.selectbox("Select the Y-axis feature:", data.columns)
            fig = px.scatter(
                data,
                x=x_feature,
                y=y_feature,
                title=f"Scatter Plot of {x_feature} vs {y_feature}",
            )
            st.plotly_chart(fig)

        elif viz_type == "Bar Chart":
            x_feature = st.selectbox("Select the X-axis feature:", data.columns)
            y_feature = st.selectbox("Select the Y-axis feature:", data.columns)
            fig = px.bar(
                data,
                x=x_feature,
                y=y_feature,
                title=f"Bar Chart of {x_feature} vs {y_feature}",
            )
            st.plotly_chart(fig)

        elif viz_type == "Line Chart":
            x_feature = st.selectbox("Select the X-axis feature:", data.columns)
            y_feature = st.selectbox("Select the Y-axis feature:", data.columns)
            fig = px.line(
                data,
                x=x_feature,
                y=y_feature,
                title=f"Line Chart of {x_feature} vs {y_feature}",
            )
            st.plotly_chart(fig)

        elif viz_type == "Box Plot":
            x_feature = st.selectbox("Select the X-axis feature:", data.columns)
            y_feature = st.selectbox("Select the Y-axis feature:", data.columns)
            fig = px.box(
                data,
                x=x_feature,
                y=y_feature,
                title=f"Box Plot of {y_feature} by {x_feature}",
            )
            st.plotly_chart(fig)

        elif viz_type == "Pie Chart":
            feature = st.selectbox(
                "Select the categorical feature for Pie Chart:",
                data.select_dtypes(include=["object", "category"]).columns,
            )
            fig = px.pie(data, names=feature, title=f"Pie Chart of {feature}")
            st.plotly_chart(fig)

        elif viz_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                st.error("No numeric data available for correlation heatmap.")
                return
            correlation = numeric_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)

    except Exception as e:
        st.error(
            f"An error occurred during visualization: {e}. Please check your data or settings."
        )
