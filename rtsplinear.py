import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to fetch stock data
def fetch_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Get company ticker input
company = input("Enter the company ticker: ")c

# Fetch historical stock data for the specified company
data = fetch_stock_data(company)

# Feature engineering
data['Date'] = data.index
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data = data.dropna()

# Define features and target variable
X = data[['Date', 'Open', 'High', 'Low', 'Volume', 'MA10', 'MA50']].values
y = data['Close'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print(f'Train R-squared: {r2_train}')
print(f'Test R-squared: {r2_test}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Actual Prices')
plt.plot(data.index[:len(X_train)], y_pred_train, label='Train Predictions', color='green')
plt.plot(data.index[len(X_train):], y_pred_test, label='Test Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{company} Stock Price Prediction using Linear Regression')
plt.legend()
plt.show()

# Real-time prediction
import datetime

def predict_future_price(days_ahead):
    future_date = datetime.date.today() + datetime.timedelta(days=days_ahead)
    future_date_ordinal = np.array([[future_date.toordinal()]])
    latest_data = data.iloc[-1]
    future_features = np.array([[future_date.toordinal(), latest_data['Open'], latest_data['High'], latest_data['Low'], latest_data['Volume'], latest_data['MA10'], latest_data['MA50']]])
    future_price = regressor.predict(future_features)
    return future_price[0]

# Predict stock price for the next day
days_ahead = 1
predicted_price = predict_future_price(days_ahead)
print(f'Predicted {company} stock price for {datetime.date.today() + datetime.timedelta(days=days_ahead)}: ${predicted_price:.2f}')
