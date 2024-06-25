# Real-Time Stock Price Prediction using Linear Regression

This project demonstrates how to predict stock prices using a linear regression model. The model is enhanced with additional features like moving averages and trading volume, and it fetches real-time data for prediction.

## Features

- Fetches real-time stock data using the `yfinance` library
- Implements a linear regression model using `scikit-learn`
- Enhances the model with additional features like moving averages (10-day and 50-day)
- Splits data into training and testing sets
- Evaluates model performance using Mean Squared Error (MSE) and R-squared metrics
- Provides real-time stock price predictions

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/himpar21/RealTimeStockLinearRegression.git
   cd RealTimeStockLinearRegression
   
2. Install required libraries:

   ```sh
    pip install yfinance pandas numpy scikit-learn matplotlib

3. Usage
   
   Run the script:

   ```sh
   python rtsplinear.py
   ```  
   OR 
   Run the streamlit file:

   ```sh
   streamlit run streamlitfile.py
   ```

5. Input the company ticker symbol when prompted:

   ```sh
      Enter the company ticker: AMZN

6. The script will fetch historical stock data, train the model, evaluate it, and display the predicted stock price for the next day.

## Code Overview

1. Fetching Stock Data
The fetch_stock_data function fetches historical stock data using the yfinance library.

   ```sh
    import yfinance as yf
    def fetch_stock_data(ticker, period='1y'):
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data

2. Feature Engineering
We convert the date to ordinal format and add moving averages (10-day and 50-day) to the dataset.
   ```sh
      data['Date'] = data.index
      data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
      data['MA10'] = data['Close'].rolling(window=10).mean()
      data['MA50'] = data['Close'].rolling(window=50).mean()
      data = data.dropna()

3. Training and Evaluating the Model
We split the data into training and testing sets, train the linear regression model, and evaluate it using MSE and R-squared metrics.**

   ```sh
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    X = data[['Date', 'Open', 'High', 'Low', 'Volume', 'MA10', 'MA50']].values
    y = data['Close'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f'Train MSE: {mse_train}')
    print(f'Test MSE: {mse_test}')
    print(f'Train R-squared: {r2_train}')
    print(f'Test R-squared: {r2_test}')

4. Real-Time Prediction
We predict the stock price for the next day using the trained model.

   ```sh
    import datetime
    def predict_future_price(days_ahead):
        future_date = datetime.date.today() + datetime.timedelta(days=days_ahead)
        future_date_ordinal = np.array([[future_date.toordinal()]])
        latest_data = data.iloc[-1]
        future_features = np.array([[future_date.toordinal(), latest_data['Open'], latest_data['High'], latest_data['Low'], latest_data['Volume'], latest_data['MA10'], latest_data['MA50']]])
        future_price = regressor.predict(future_features)
        return future_price[0]
    
    days_ahead = 1
    predicted_price = predict_future_price(days_ahead)
    print(f'Predicted {company} stock price for {datetime.date.today() + datetime.timedelta(days=days_ahead)}: ${predicted_price:.2f}')
   
## Results
The script will display the actual and predicted stock prices and plot them for better visualization.
   ```sh
   Train MSE: 0.6011100765763535
   Test MSE: 1.2201640730661958
   Train R-squared: 0.998634006149385
   Test R-squared: 0.9964225315968204
   Predicted AMZN stock price for 2024-06-25: $187.77
```
![Figure_1](https://github.com/himpar21/RealTimeStockLinearRegression/assets/95409033/bae84959-a889-40f3-ab7b-2b956cf20703)


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
