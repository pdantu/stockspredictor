import pandas as pd
import yfinance as yf
from random import random
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import stocker
def main():
    predict()
    onedaypred('SPY')
    #getNextDaypred(['SPY'])

def getNextDaypred(etfs):
    for x in etfs:
        a = yf.Ticker(x)
        prices = a.history('max')
        prices = prices.reset_index()

        if prices.shape[0] == 0:
            return prices
        prices = prices.iloc[:-1]
        model = ExponentialSmoothing(prices['Close'])
        model_fit = model.fit()
        yhat = model_fit.predict(len(prices), len(prices))
        
        print(yhat)
        prices['1DayEMA'] = prices['Open'].ewm(span = 1).mean()
        prices['2DayEMA'] = prices['Open'].ewm(span = 2).mean()
        prices = prices[['Date', 'Open', 'Close', '1DayEMA', '2DayEMA']]
        print(prices.tail())
        if prices['1DayEMA'].iloc[len(prices) - 1] > prices['2DayEMA'].iloc[len(prices) - 1]:
            return(2)
        else:
            return(1)
def predict():
    
    pd.options.mode.chained_assignment = None
    tf.random.set_seed(0)

    # download the data
    df = yf.download(tickers=['SPY'], period='1y')
    y = df['Close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 1  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Open', 'Close','Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('Date')
    print(results.tail(2))
    # plot the results
    #results.plot(title='AAPL')

def onedaypred(etf):
    
    print(stocker.predict.tomorrow('SPY'))

    """
    a = yf.Ticker('SPY')
    df = a.download('max')
    y = df[:, 'Close'].as_matrix()
    train_data = y[:11000]
    test_data = y[11000:]
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)"""

if __name__ == "__main__":
    main()