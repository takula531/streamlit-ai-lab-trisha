"""
Streamlit Exploratory Data Analysis App
Loads the Iris dataset and provides interactive visualization and analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Iris EDA App",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Exploratory Data Analysis - Iris Dataset")
st.markdown("Analyze the Iris dataset with interactive visualizations and statistics")

# Load the Iris dataset
@st.cache_data
def load_iris_data():
    """Load and convert the Iris dataset to a pandas DataFrame."""
    iris_dataset = datasets.load_iris()
    iris_df = pd.DataFrame(
        data=iris_dataset.data,
        columns=iris_dataset.feature_names
    )
    iris_df['Species'] = pd.Categorical.from_codes(
        iris_dataset.target,
        iris_dataset.target_names
    )
    return iris_df

# Load the data
iris_data = load_iris_data()

# Display section: Dataset Overview
st.header("📋 Dataset Overview")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Shape")
    st.info(f"Rows: {iris_data.shape[0]} | Columns: {iris_data.shape[1]}")

with col2:
    st.subheader("Column Names")
    st.write(", ".join(iris_data.columns.tolist()))

# Display first few rows
st.subheader("First Rows of Data")
num_rows = st.slider(
    "Number of rows to display:",
    min_value=1,
    max_value=20,
    value=5
)
st.dataframe(iris_data.head(num_rows), use_container_width=True)

# Display section: Summary Statistics
st.header("📈 Summary Statistics")
st.subheader("Descriptive Statistics")
st.dataframe(iris_data.describe(), use_container_width=True)

# Display section: Visualizations
st.header("📊 Visualizations")

# Get numeric columns for analysis
numeric_columns = iris_data.select_dtypes(include=[np.number]).columns.tolist()

# Histogram section
st.subheader("Histogram Analysis")
selected_hist_column = st.selectbox(
    "Select a numeric column for histogram:",
    numeric_columns,
    key="histogram_column"
)

# Create and display histogram
fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
ax_hist.hist(iris_data[selected_hist_column], bins=20, color='steelblue', edgecolor='black')
ax_hist.set_xlabel(selected_hist_column)
ax_hist.set_ylabel("Frequency")
ax_hist.set_title(f"Distribution of {selected_hist_column}")
ax_hist.grid(axis='y', alpha=0.3)
st.pyplot(fig_hist)

# Scatter plot section
st.subheader("Scatter Plot Analysis")

# Create two columns for selecting x and y axes
scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    selected_x_column = st.selectbox(
        "Select X-axis column:",
        numeric_columns,
        key="scatter_x"
    )

with scatter_col2:
    selected_y_column = st.selectbox(
        "Select Y-axis column:",
        numeric_columns,
        index=1,
        key="scatter_y"
    )

# Create and display scatter plot
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))

# Plot points colored by species
for species in iris_data['Species'].unique():
    species_data = iris_data[iris_data['Species'] == species]
    ax_scatter.scatter(
        species_data[selected_x_column],
        species_data[selected_y_column],
        label=species,
        alpha=0.7,
        s=100
    )

ax_scatter.set_xlabel(selected_x_column)
ax_scatter.set_ylabel(selected_y_column)
ax_scatter.set_title(f"{selected_x_column} vs {selected_y_column}")
ax_scatter.legend()
ax_scatter.grid(True, alpha=0.3)
st.pyplot(fig_scatter)

# Display section: Species Information
st.header("🌸 Species Distribution")
species_count = iris_data['Species'].value_counts()
st.bar_chart(species_count)

st.write(f"**Species Count:**")
for species, count in species_count.items():
    st.write(f"- {species}: {count} samples")
