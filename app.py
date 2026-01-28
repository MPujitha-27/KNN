import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Page config
st.set_page_config(page_title="KNN Classification", layout="centered")

# Title
st.title("🔍 KNN Classification using Streamlit")
st.write("Interactive frontend for K-Nearest Neighbors Classification")

# Sidebar controls
st.sidebar.header("⚙️ Model Parameters")

n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000)
n_features = st.sidebar.slider("Number of features", 2, 10, 3)
k_value = st.sidebar.slider("K (No. of Neighbors)", 1, 15, 5)
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)

# Generate dataset
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
st.subheader("📊 Model Performance")
st.success(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

st.subheader("🔲 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Classification Report
st.subheader("📄 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report)

# Footer
st.markdown("---")

