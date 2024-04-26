import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Generate synthetic stock price data
np.random.seed(0)
dates = pd.date_range(start='2020-01-01', end='2022-01-01', freq='B')
prices = np.random.normal(loc=100, scale=10, size=len(dates))

data = pd.DataFrame({'Date': dates, 'Close': prices})

# Save data to CSV and reload
data.to_csv('historical_stock_prices.csv', index=False)

data = pd.read_csv('historical_stock_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Sidebar title and description
st.sidebar.title('Stock Price Analysis')
st.sidebar.write('Performing time series analysis using ARIMA')

# Display basic info in sidebar
st.sidebar.write('Data loaded successfully!')
st.sidebar.write(data.head())

# Plot rolling mean and standard deviation
rolling_mean = data['Close'].rolling(window=30).mean()
rolling_std = data['Close'].rolling(window=30).std()

st.write('## Rolling Mean & Standard Deviation')
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='green', label='Rolling Std')
plt.legend()
plt.title('Rolling Mean & Standard Deviation')
st.pyplot(plt)

# Plot ACF and PACF
st.write('## Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)')
fig, ax = plt.subplots(nrows=2, figsize=(12, 8))
plot_acf(data['Close'], lags=20, ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(data['Close'], lags=20, ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
st.pyplot(fig)

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Fit ARIMA model
model = ARIMA(train_data['Close'], order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')

# Evaluate model
mse = mean_squared_error(test_data['Close'], predictions)
st.write('## ARIMA Model Evaluation')
st.write('Mean Squared Error:', mse)

# Plot predictions
st.write('## ARIMA Model Forecast')
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Close'], label='Train')
plt.plot(test_data.index, test_data['Close'], label='Test')
plt.plot(test_data.index, predictions, color='red', label='Predictions')
plt.legend()
plt.title('ARIMA Model Forecast')
st.pyplot(plt)
