import streamlit as st
import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt


def plot_shap_values(
    model: xgb.XGBModel,
    X: pd.DataFrame,
    feature_names: list,
    plot_type: str,
    dep_feature: str = None,
) -> None:
    """
    Plot SHAP values for model interpretability.

    Args:
        model (xgb.XGBRegressor): The trained XGBoost model.
        X (pd.DataFrame): The feature data.
        feature_names (list): List of feature names.
        plot_type (str): Type of SHAP plot to generate.
        dep_feature (str, optional): Feature for dependence plot.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if plot_type == "Summary Plot":
            st.subheader("SHAP Summary Plot")
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            fig = plt.gcf()  # Get the current figure
            st.pyplot(fig)

        elif plot_type == "Dependence Plot":
            if dep_feature is not None and dep_feature in feature_names:
                st.subheader(f"SHAP Dependence Plot for {dep_feature}")
                shap.dependence_plot(
                    dep_feature, shap_values, X, feature_names=feature_names, show=False
                )
                fig = plt.gcf()  # Get the current figure
                st.pyplot(fig)
            else:
                st.error("Please select a valid feature for the dependence plot.")

        elif plot_type == "Waterfall Plot":
            st.subheader("SHAP Waterfall Plot")
            instance_idx = st.slider(
                "Select instance index for Waterfall Plot", 0, X.shape[0] - 1, 0
            )
            try:
                explanation = shap.Explanation(
                    values=shap_values[instance_idx],
                    base_values=explainer.expected_value,
                    data=X.iloc[instance_idx],
                    feature_names=feature_names,
                )
                shap.waterfall_plot(explanation, show=False)
                fig = plt.gcf()  # Get the current figure
                st.pyplot(fig)
            except ValueError:
                st.error(
                    "Waterfall plot error: This plot may not be supported for this data."
                )

    except Exception as e:
        st.error(f"An error occurred during SHAP analysis: {e}")
