import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files for TSEM and UMC.
# The 'thousands' parameter ensures that numbers like "14,390" are correctly interpreted.
tsem = pd.read_csv("TSEM_1.csv", thousands=',')
umc  = pd.read_csv("ONS_1.csv", thousands=',')

# Convert the Date columns to datetime objects (assumes MM/DD/YYYY format)
tsem['Date'] = pd.to_datetime(tsem['Date'], format='%m/%d/%Y')
umc['Date']  = pd.to_datetime(umc['Date'], format='%m/%d/%Y')

# Merge the two dataframes on the Date column.
# We assume that both files have a 'Price' column.
df = pd.merge(
    tsem[['Date', 'Price']],
    umc[['Date', 'Price']],
    on='Date',
    suffixes=('_TSEM', '_UMC')
)

# Convert the Price columns to float and rename them for clarity.
df['TSEM'] = df['Price_TSEM'].astype(float)
df['UMC'] = df['Price_UMC'].astype(float)

# Optionally, sort the data by Date.
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Compute the spread between TSEM and UMC prices.
df['Spread'] = df['TSEM'] - df['UMC']

# Create a new DataFrame (or select the relevant columns) for correlation analysis.
analysis_df = df[['TSEM', 'UMC', 'Spread']]

# Calculate the correlation matrix.
corr_matrix = analysis_df.corr()

# Display the correlation matrix.
print(corr_matrix)

# Plot the correlation matrix as a heatmap.
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for TSEM, ONS, and Spread')
plt.show()
