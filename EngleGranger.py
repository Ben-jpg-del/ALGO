import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt

# Read CSV files for TSEM and UMC.
# The CSV files have a Price column with commas as thousands separators.
tsem = pd.read_csv("GLD.csv", thousands=',')
umc  = pd.read_csv("GDX.csv", thousands=',')

# Convert the Date column to datetime objects (assuming MM/DD/YYYY format)
tsem['Date'] = pd.to_datetime(tsem['Date'], format='%m/%d/%Y')
umc['Date']  = pd.to_datetime(umc['Date'], format='%m/%d/%Y')

# Merge the two dataframes on the Date column.
# We assume both files contain a 'Price' column.
data = pd.merge(
    tsem[['Date', 'Price']],
    umc[['Date', 'Price']],
    on='Date',
    suffixes=('_tsem', '_umc')
)

# Rename the price columns to asset1 and asset2 for consistency.
# Here asset1 corresponds to TSEM and asset2 corresponds to UMC.
# (If needed, ensure the Price columns are floats.)
data['asset1'] = data['Price_tsem'].astype(float)
data['asset2'] = data['Price_umc'].astype(float)

# Optional: sort the data by date
data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)

# Function to perform the ADF test and print the results.
def adf_test(series, name=''):
    result = adfuller(series)
    print(f'ADF Test for {name}:')
    print(f"ADF Statistic for {name}: {result[0]}")
    print(f"p-value for {name}: {result[1]}")
    for key, value in result[4].items():
        print(f"Critical Value {key}: {value}")
    return result[1]

# Step 1: Test each series for stationarity
print("Step 1: Test each series for stationarity")
adf_test(data['asset1'], 'asset1')
adf_test(data['asset2'], 'asset2')

# Step 2: Perform the Engle-Granger cointegration test
print("\nStep 2: Perform the Engle-Granger cointegration test")
score, pvalue, _ = coint(data['asset1'], data['asset2'])
print(f"Engle-Granger Cointegration Test score: {score}")
print(f"Engle-Granger Cointegration Test p-value: {pvalue}")

# Step 3: Visualize the series and their spread
data['spread'] = data['asset1'] - data['asset2']

plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(data['Date'], data['asset1'], label='Asset 1 (TSEM)')
plt.plot(data['Date'], data['asset2'], label='Asset 2 (UMC)')
plt.legend()
plt.title('Non-Stationary Time Series')

plt.subplot(2, 1, 2)
plt.plot(data['Date'], data['spread'], label='Spread (Asset 1 - Asset 2)')
plt.legend()
plt.title('Spread (Should be Stationary)')

plt.tight_layout()
plt.show()

# Step 4: Test the spread for stationarity
print("\nStep 4: Test the spread for stationarity")
adf_test(data['spread'], 'spread')
