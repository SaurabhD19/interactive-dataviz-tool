import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from auth import login_page, register_page
from visualization import plot_data_visualization
from shap_analysis import plot_shap_values
from config import SHAP_PLOT_TYPES


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        choice = st.sidebar.selectbox("Login/Signup", ["Login", "Register"])
        if choice == "Login":
            login_page()
        elif choice == "Register":
            register_page()
    else:
        page = st.sidebar.selectbox(
            "Select Page", ["Data Visualization", "SHAP Analysis", "Logout"]
        )
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:", data.head())
                st.subheader("Select Target Column")
                potential_targets = [
                    col for col in data.columns if data[col].nunique() < 10
                ]
                st.write("Suggested Target Columns:", potential_targets)
                target = st.selectbox("Choose the target column:", data.columns)

                X = data.drop(columns=[target])
                y = data[target]
                X = pd.get_dummies(X, drop_first=True)
                for col in X.columns:
                    strategy = "most_frequent" if X[col].dtype == "object" else "median"
                    imputer = SimpleImputer(strategy=strategy)
                    X[col] = imputer.fit_transform(X[[col]])
                if X.isnull().sum().any():
                    st.error("Please clean your data. Missing values remain.")
                    return
                if y.dtype == "object" or len(np.unique(y)) <= 10:
                    y = LabelEncoder().fit_transform(y)
                scaler = StandardScaler()
                X = pd.DataFrame(
                    scaler.fit_transform(X), columns=X.columns
                )  # Preserve column names

                if page == "SHAP Analysis":
                    # Determine task type based on target variable
                    is_classification = data[target].nunique() <= 10 or data[target].dtype == "object"
                    
                    if is_classification:
                        st.info("Task detected as **Classification**.")
                        model = xgb.XGBClassifier().fit(X, y)
                        preds = model.predict(X)
                        accuracy = accuracy_score(y, preds)
                        st.success(f"Model Training Accuracy: **{accuracy:.2f}**")
                    else:
                        st.info("Task detected as **Regression**.")
                        model = xgb.XGBRegressor().fit(X, y)
                        preds = model.predict(X)
                        r2 = r2_score(y, preds)
                        mse = mean_squared_error(y, preds)
                        st.success(f"Model Training R²: **{r2:.2f}**")
                        st.write(f"Model Training MSE: **{mse:.2f}**")

                    plot_type = st.selectbox("Select SHAP Plot Type", SHAP_PLOT_TYPES)
                    dep_feature = (
                        st.selectbox("Select feature for Dependence Plot", X.columns)
                        if plot_type == "Dependence Plot"
                        else None
                    )
                    plot_shap_values(model, X, X.columns, plot_type, dep_feature)
                elif page == "Data Visualization":
                    plot_data_visualization(data, target)

            except Exception as e:
                st.error(
                    f"An error occurred while processing the file: {e}. Please check your file format and data."
                )
        if page == "Logout":
            st.session_state["logged_in"] = False
            st.success("You have logged out.")
            st.rerun()


if __name__ == "__main__":
    main()
