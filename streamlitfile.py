import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Streamlit app
st.title("Enhanced Real-Time Stock Price Prediction using Linear Regression")

# User input for ticker symbol
ticker = st.text_input("Enter stock ticker symbol:", "AAPL")

if ticker:
    try:
        # Fetch historical stock data
        data = yf.download(ticker, period="2y")

        if not data.empty:
            # Feature Engineering: Add moving averages
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data = data.dropna()

            # Preprocess data
            data.loc[:, 'Date'] = data.index
            data.loc[:, 'Date'] = pd.to_datetime(data['Date'])
            data.loc[:, 'Date'] = data['Date'].map(pd.Timestamp.timestamp)

            # Define features and target variable
            X = data[['Date', 'MA5', 'MA20']].values
            y = data['Close'].values

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate accuracy (mean squared error and R-squared)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Model Mean Squared Error: {mse:.2f}")
            st.write(f"Model R-squared: {r2:.2f}")

            # Predict the next day's price
            next_day_timestamp = np.array([[X[-1][0] + 86400, data['MA5'].values[-1], data['MA20'].values[-1]]])
            next_day_price = model.predict(next_day_timestamp)[0]
            st.write(f"Predicted price for next day: ${next_day_price:.2f}")

            # Plot actual vs predicted prices
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, y, label='Actual Prices')
            ax.plot(data.index[len(data)-len(y_test):], y_pred, color='red', label='Predicted Prices')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'{ticker} Stock Price Prediction')
            ax.legend()
            st.pyplot(fig)

        else:
            st.write("Invalid ticker symbol or no data available.")
    except Exception as e:
        st.write(f"An error occurred: {e}")
