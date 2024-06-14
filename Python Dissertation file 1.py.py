#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset into a DataFrame
df = pd. read_csv('Car_SupplyChainManagementDataSet.csv')


# In[2]:


df


# In[3]:


# Step 3: Inspect the dataset
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)


# In[22]:


# Step 4: Data Cleaning and Preprocessing
# Remove unnecessary columns (if any)
df = df.drop(columns=["SupplierContactDetails"])

# Convert date columns to datetime
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["ShipDate"] = pd.to_datetime(df["ShipDate"])

# Remove duplicates (if any)
df = df.drop_duplicates()

# Replace missing values with appropriate values or strategies (e.g., mean, median)
df["Discount"].fillna(0, inplace=True)

# Convert columns to appropriate data types
df["Quantity"] = df["Quantity"].astype(int)

# Remove any leading/trailing spaces in string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


# In[5]:


# Step 5: Save the cleaned dataset to a new CSV file
df.to_csv("Cleaned_Car_SupplyChainManagementDataSet.csv", index=False)

# Display the first few rows of the cleaned dataset
print("Cleaned Dataset:")
print(df.head())


# In[6]:


# Display basic information about the dataset
print("Number of rows and columns:", df.shape)


# In[7]:


print("\nColumn names:", df.columns)


# In[8]:


print("\nData types:\n", df.dtypes)


# In[9]:


print("\nSummary statistics:\n", df.describe())


# In[10]:


# Check for missing values
print("\nMissing values:\n", df.isnull().sum())


# In[11]:


# Check for duplicated rows
duplicates = df[df.duplicated()]
print("\nNumber of duplicated rows:", duplicates.shape[0])


# In[12]:


# Data Visualization

# Plot histograms for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# In[17]:


# Pairwise scatter plots for numeric columns
sns.pairplot(df[numeric_cols])
plt.suptitle("Pairwise Scatter Plots for Numeric Columns", y=1.02)
plt.show()


# In[14]:


# Box plots to identify outliers in numeric columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Box Plot of {col}")
    plt.xlabel(col)
    plt.show()


# In[15]:


# Correlation matrix for all columns
all_corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(all_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Including Categorical)")
plt.show()


# In[ ]:




