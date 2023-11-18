import streamlit as st
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import mode
import statistics
import scipy.stats as ss

# add a title to the webapp
st.title("Automatic Basic Exploratory Data Analysis (EDA)")
# add a sidebar for the file uploader
st.sidebar.subheader("Visualization Settings")
# select the placeholder for file uploader
file_csv = st.sidebar.file_uploader(label="Upload your CSV file here", type="csv")

if file_csv is None:
    st.subheader("Upload Your Dataframe From the Sidebar Menu")
else:
    df = pd.read_csv(file_csv)
    st.subheader("You have uploaded the following Dataframe")
    st.dataframe(data=df)
    shape = str(df.shape)
    st.write(f"DataFrame Shape: {shape}")
    # pr = df.profile_report()
    # st_profile_report(pr)
    st.subheader("Select Only the Relevant Columns for EDA")
    sel_cols = st.multiselect("Select Relevant Columns: ", df.columns)
    st.write(f"Selected Columns: {sel_cols}")
    new_df = df[sel_cols]
    st.write("Selected New DataFrame")
    st.dataframe(new_df)
    new_shape = str(new_df.shape)
    st.write(f"DataFrame Shape: {new_shape}")

    # setting the target column and independent column
    target_cols = st.selectbox("Target Column: ", new_df.columns)
    st.write(f"Target Column is {target_cols}")
    indt_cols = new_df.drop(columns=target_cols, axis=1)
    st.write(f"Independent Columns are {indt_cols.columns}")

    # visualizing the missing Values
    st.subheader("Data Processing")
    st.write("Missing Values Visualization")
    fig, ax = plt.subplots()
    msno.matrix(new_df, ax=ax)
    st.pyplot(fig)

    # Categorizing the Columns
    cat_cols = []
    num_cols = []
    for column in new_df.columns:
        if new_df[column].dtypes == object:
            cat_cols.append(column)
        else:
            num_cols.append(column)

    # Missing Values Imputation
    for column in cat_cols:
        values, counts = np.unique(new_df[column].dropna(), return_counts=True)
        mode_value = values[np.argmax(counts)]
        new_df[column] = new_df[column].fillna(mode_value)
    for column in num_cols:
        if new_df[column].dtypes == int:
            new_df[column] = new_df[column].fillna(mode(new_df[column])[0])
        else:
            new_df[column] = new_df[column].fillna(np.mean(new_df[column]))

    # Visualizing Again Missing Values
    st.write("Missing Values Visualization After Filling Missing Values")
    fig, ax = plt.subplots()
    msno.matrix(new_df, ax=ax)
    st.pyplot(fig)

    # Checking the pearson Correlation for Numeric Data

    st.subheader("Pearson Correlation Matrix for Numeric Data")
    corr_matrix = new_df[num_cols].corr()
    st.write(corr_matrix)
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt)

    # Checking the Association Between Categorical Columns using Cramer'V Test Based on Chi-Squred Principle
    st.subheader("Association for Categorical Data")

    def cramers_v(x, y):
        contingency_table = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def association(df):
        target = target_cols
        association_results = []
        for column in new_df[cat_cols].columns:
            if column != target:
                association = cramers_v(new_df[target], new_df[column])
                association_results.append((column, association))
        association_df = pd.DataFrame(association_results, columns=["Features", "Cramer's V"])
        association_df = association_df.sort_values(by=["Cramer's V"], ascending=False)
        association_df = association_df.reset_index(drop=True)
        return association_df

    st.dataframe(association(new_df))
    # Pair Plot
    st.subheader("Data Visualization")
    plt.figure(figsize=(12, 12))
    sns.pairplot(new_df[num_cols])
    st.pyplot(plt)
    # Box Plot
    plt.figure(figsize=(12, 20))
    sns.boxplot(new_df[num_cols])
    st.pyplot(plt)

    # Display the cleaned DataFrame
    st.write("Cleaned DataFrame:")
    st.write(new_df)

    # Add a download button for the cleaned DataFrame
    st.download_button(
        label="Download Cleaned DataFrame",
        data=new_df.to_csv().encode('utf-8'),
        file_name='cleaned_dataframe.csv',
        key='download_cleaned_dataframe'
    )






