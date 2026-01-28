import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Page config
st.set_page_config(page_title="KNN Regression", layout="centered")

# Title
st.title("📈 KNN Regression using Streamlit")
st.write("Interactive frontend for K-Nearest Neighbors Regression")

# Sidebar parameters
st.sidebar.header("⚙️ Model Parameters")

n_samples = st.sidebar.slider("Number of samples", 100, 3000, 1000)
n_features = st.sidebar.slider("Number of features", 1, 10, 3)
k_value = st.sidebar.slider("K (No. of Neighbors)", 1, 20, 5)
noise = st.sidebar.slider("Noise", 0, 50, 10)
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)

# Generate dataset
X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    noise=noise,
    random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)

# Train model
model = KNeighborsRegressor(n_neighbors=k_value)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display metrics
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("MAE", f"{mae:.2f}")

# Actual vs Predicted plot
st.subheader("📉 Actual vs Predicted")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Built using **Streamlit + Scikit-learn**")
