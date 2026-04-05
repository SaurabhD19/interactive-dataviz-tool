# 📊 End-to-End ML & Interpretability Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-blue?logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)

A dynamic, fully interactive web application built with **Streamlit** that allows users to seamlessly upload data, generate rich visual insights, and automatically train Machine Learning models equipped with **SHAP (SHapley Additive exPlanations)** for Explainable AI (XAI).

---

## ✨ Key Features

- 🔐 **Secure User Authentication:** Built-in login and registration functionality to control dashboard access.
- ⚙️ **Automated ML Preprocessing:** Automatically handles missing data imputation, standard scales numeric features, and safely label-encodes categorical text.
- 🧠 **Smart Dynamic Modeling:** Intelligently detects whether your dataset requires **Classification** or **Regression**, and trains the correct XGBoost model accordingly.
- 📈 **Interactive Exploratory Data Analysis (EDA):** Generate histograms, scatter plots, box plots, pie charts, and correlation heatmaps dynamically using Plotly and Seaborn.
- 🕵️‍♂️ **Explainable AI (SHAP):** Unbox your model's decisions! Visualize feature importance using interactive SHAP Summary, Dependence, and Waterfall plots.

---

## 📂 Best Dataset Examples to Try

The app is highly flexible. For the best experience, try downloading these classic **Kaggle** datasets in `.csv` format and uploading them:

*   🚢 **Titanic Survival Dataset** *(Classification)* - Predict who survived based on Age, Cabin, and Fare. Great for Pie Charts and SHAP feature importance.
*   🏡 **Boston/California Housing Prices** *(Regression)* - Predict house prices based on rooms, taxes, and location. Perfect for Correlation Heatmaps and Scatter Plots.
*   🌸 **Iris Flower Dataset** *(Classification)* - A classic, clean dataset to predict flower species based on petal dimensions.
*   🩺 **Heart Disease Dataset** *(Classification)* - Predict the presence of heart disease using patient health metrics.

---

## 🚀 Setup & Installation Instructions

Follow these steps to get the application running on your local machine:

1. **Clone the repository (or extract the project folder):**
   ```bash
   git clone https://github.com/SaurabhD19/interactive-dataviz-tool.git
   cd interactive-dataviz-tool
   ```

2. **Create a virtual environment (Highly Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   Ensure you have all required libraries installed.
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application:**
   Start the Streamlit server.
   ```bash
   streamlit run Viz.py
   ```
   *The app will automatically open in your default web browser at `http://localhost:8501`*

---

## 🎮 How to Use the App

1. **Register/Login:** Create a quick mock account on the sidebar to access the hidden dashboard pages.
2. **Upload Data:** Upload any clean `.csv` dataset (ensure the file is correctly formatted with headers).
3. **Select Target:** Choose the column you want the AI to learn to predict. The app will let you know if it is treating your target as a Classification or Regression problem.
4. **Data Visualization:** Select 'Data Visualization' from the menu and switch between Plotly charts to find correlations and data outliers.
5. **SHAP Analysis:** Select 'SHAP Analysis' from the menu. Evaluate the model's live training performance (Accuracy or R²/MSE) and interact with SHAP plots to understand *how* the model is making its predictions.

---