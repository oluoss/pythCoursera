# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the dataset
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df = pd.read_csv(URL)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(boston_df.head())

# Get summary statistics of the dataset
print("\nSummary Statistics:")
print(boston_df.describe())

# Check for missing values
print("\nMissing Values:")
print(boston_df.isnull().sum())

# Visualize correlation using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(boston_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Boston Housing Data")
plt.show()

# Perform a regression analysis
model = ols("MEDV ~ CRIM + RM + PTRATIO", data=boston_df).fit()

# Print regression summary
print("\nRegression Summary:")
print(model.summary())

# Scatter plot of RM vs MEDV
plt.figure(figsize=(8, 6))
sns.scatterplot(x=boston_df["RM"], y=boston_df["MEDV"], alpha=0.7)
plt.xlabel("Average Number of Rooms per Dwelling (RM)")
plt.ylabel("Median Value of Owner-Occupied Homes (MEDV)")
plt.title("Scatter Plot: RM vs MEDV")
plt.show()
