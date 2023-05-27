from fredapi import Fred
import pandas as pd
import yfinance as yf
import os
# pfrom statsmodels.tsa.arima_model import ARIMA

path = os.getcwd()

fred = Fred(api_key='13f1e0b5cbdcb8307bbf7bbca9852e4a')

def main():
    data = {'GDP': 'GDP', 'UNrate': 'UnemploymentRate', 'GDPC1': 'RealGDP', 'SP500': 'MarketPrice', 'DGDSRX1Q020SBEA': 'PersonalConsumption', 'IMPGSC1': 'Imports', 'NETEXC': 'Exports', 'DFF': 'FedRateHike', 'CORESTICKM159SFRBATL': 'CPI', 'UMCSENT': 'ConsumerSentiment'}
    # getData(data)
    # combineFredData(data)
    #df = pd.read_csv(path + '/macroecondata/GDP.csv')
    #getPredictionGDP(df)
    # getSpySectorWeights()
    #mergeData()
    calculateMarketScore()
 
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