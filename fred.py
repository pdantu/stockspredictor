from fredapi import Fred
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import numpy as np
# pfrom statsmodels.tsa.arima_model import ARIMA

path = os.getcwd()

fred = Fred(api_key='13f1e0b5cbdcb8307bbf7bbca9852e4a')

def main():
    data = {'GDP': 'GDP', 'UNrate': 'UnemploymentRate', 'GDPC1': 'RealGDP', 'SP500': 'MarketPrice', 'DGDSRX1Q020SBEA': 'PersonalConsumption', 'IMPGSC1': 'Imports', 'NETEXC': 'Exports', 'DFF': 'FedRateHike', 'CORESTICKM159SFRBATL': 'CPI', 'UMCSENT': 'ConsumerSentiment'}
    # getData(data)
    # combineFredData(data)
    df = pd.read_csv(path + '/macroecondata/final.csv')
    #getPredictionGDP(df)
    # getSpySectorWeights()
    #mergeData()
    # calculateMarketScore()
    predict(df['GDP'])
 
def getData(values):
    for x in values:
        data = fred.get_series_latest_release(x)
        data = data.reset_index()
        data.columns = ['Date',  values[x]]
        data.to_csv(path + '/macroecondata/' + values[x] + '.csv')
#categories = ['UNrate', 'GDP', 'SP500','CPALTT01USM657N' , FPCPITOTLZGUSA (inflation), GDPC1 (real gdp), IMPGSC1 (real imports of goods), NETEXC (real exports), DGDSRX1Q020SBEA (personal consumption of goods), CORESTICKM159SFRBATL (CPI), DFF (Fed rate hikes), UMCSENT (consumer sentiment), RSXFS (retail sales), PPIACO (Producer price index), RECPROUSM156N (smoothed US rececssion probability)]

def combineFredData(values):
    data = pd.read_csv(path + '/macroecondata/GDP.csv')
    # print(data.columns)
    for x in values:
        if values[x] == 'GDP':
            continue
        df = pd.read_csv(path + '/macroecondata/' + values[x] + '.csv')
        data = data.merge(df, on = 'Date', how = 'left')
        
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.to_csv(path + '/macroecondata/final.csv')


def calculateMarketScore():
    data = pd.read_csv(path + '/macroecondata/final.csv')
    data = data.drop(['MarketPrice', 'PersonalConsumption'], axis = 1)
    c = 0
    norm = {}
    print(data.shape)
    data = data.dropna()
    weights = {'GDP': 0.2, 'UnemploymentRate': 0.15, 'CPI': 0.12, 'PersonalConsumption': .12, 'FedRateHike': .1, 'ConsumerSentiment': 0.1, 'RealGDP': 0.08, 'Imports': 0.1, 'Exports': 0.1}
    for x in data.columns:
        if c < 2:
            c += 1
            continue
        norm[x] = (data[x].mean(), data[x].std())
    scores = []
    for index, row in data.iterrows():
        score = applyRow(row, norm, data.columns, weights)
        score = score * 100
        scores.append(score)
    data['Score'] = scores
    diff = []
    diff.append(0)
    for i in range(1, len(scores)):
        diff.append(scores[i] - scores[i-1])
    data['Difference'] = diff
    print(data.shape)
    data.to_csv(path + '/macroecondata/scores.csv')
    
def applyRow(row, norm, cols, weights):
    score = 0
    for x in cols:
        val = 0
        if x not in norm:
            continue
        val = (row[x] - norm[x][0]) / norm[x][1]
        val = val * weights[x]
        score += val
    return score

def plots(col):
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf

    # Assuming you have a pandas DataFrame called 'data' with a column named 'TimeSeriesData'
    # Ensure the 'TimeSeriesData' column is in the appropriate format (e.g., numeric)

    # Plot the PACF
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(col, lags=20, ax=ax)  # Specify the number of lags to display
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Autocorrelation')
    ax.set_title('Partial Autocorrelation Function (PACF)')
    plt.show()
    # Assuming you have a pandas DataFrame called 'data' with a column named 'TimeSeriesData'
    # Ensure the 'TimeSeriesData' column is in the appropriate format (e.g., numeric)

    # Plot the ACF
    # fig, ax = plt.subplots(figsize=(10, 5))
    # plot_acf(col, lags=20, ax=ax)  # Specify the number of lags to display
    # ax.set_xlabel('Lag')
    # ax.set_ylabel('Autocorrelation')
    # ax.set_title('Autocorrelation Function (ACF)')
    # plt.show()

def predict(col):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    import matplotlib.pyplot as plt

    # Assuming you have a pandas DataFrame called 'data' with a column named 'TimeSeriesData'
    # Ensure the 'TimeSeriesData' column is in the appropriate format (e.g., numeric)

    # Convert the time series data into a numpy array
    col = col.dropna()
    data_array = col.values.reshape(-1, 1)

    # Perform data normalization using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_array)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

    # Define the number of previous time steps to consider for each prediction
    n_steps = 10

    # Create input sequences and corresponding target values
    def create_sequences(data, n_steps):
        X = []
        y = []
        for i in range(n_steps, len(data)):
            X.append(data[i - n_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    train_X, train_y = create_sequences(train_data, n_steps)
    test_X, test_y = create_sequences(test_data, n_steps)

    # Reshape the input sequences to fit the LSTM input shape
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    model.fit(train_X, train_y, epochs=10, batch_size=32)

    # Make predictions on the test data
    predictions = model.predict(test_X)

    # Inverse transform the scaled predictions and actual values to their original scale
    predictions = scaler.inverse_transform(predictions)
    actual_values = scaler.inverse_transform(test_y.reshape(-1, 1))
    print(test_X[-1])
    # Plot the predicted values and actual values
    # plt.plot(predictions, label='Predicted')
    # plt.plot(actual_values, label='Actual')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()
    # return model

def mergeData():
    a = yf.Ticker('SPY')
    spy = a.history('max')
    spy = spy.reset_index()
    #print(type(spy['Date'].iloc[0]))
    gdp = pd.read_csv(path + '/macroecondata/GDP.csv')
    realgdp = pd.read_csv(path + '/macroecondata/RealGDP.csv')
    realgdp['Date'] = pd.to_datetime(realgdp['Date'])
    
    #print(type(realgdp['Date'].iloc[0]))
    unemploymentrate = pd.read_csv(path + '/macroecondata/UnemploymentRate.csv')

    finaldf = spy.merge(realgdp, how='inner', on='Date')
    print(finaldf.head())

def getPredictionGDP(data):
    data = data.dropna()
    model=ARIMA(data['GDP'],order=(1,1,1))
    model_fit=model.fit()
    y_pred = model_fit.get_forecast(len(data.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = model_fit.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = data.index
    y_pred_out = y_pred_df["Predictions"] 
    #print(model_fit.summary())
    #data['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
    print(y_pred_out.tail())
def getSpySectorWeights():
    a = yf.Ticker('SPY')
    vals = a.stats()['topHoldings']['sectorWeightings']
    df = pd.DataFrame(columns=['sector', 'ETF', 'weight'])
    dict = {'basic_materials': 'XLB', 'utilities': 'XLU', 'realestate': 'XLRE', 'energy': 'XLE', 'consumer_defensive': 'XLP', 'industrials': 'XLI', 'communication_services': 'XLC', 'consumer_cyclical': 'XLY', 'financial_services': 'XLF','healthcare': 'XLV','technology': 'XLK'}
    for x in vals:
        for y in x:
            
            df.loc[len(df.index)] = [y, dict.get(y), x[y]]
    
    df.to_csv(path + '/holdings/spysectorweights.csv')


if __name__ == "__main__":
    main()