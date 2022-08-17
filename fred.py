from fredapi import Fred
import pandas as pd
import yfinance as yf
import os
from statsmodels.tsa.arima_model import ARIMA

path = os.getcwd()

fred = Fred(api_key='13f1e0b5cbdcb8307bbf7bbca9852e4a')

def main():
    getData({'GDP': 'GDP', 'UNrate': 'UnemploymentRate', 'GDPC1': 'RealGDP', 'SP500': 'MarketPrice'})
    df = pd.read_csv(path + '/macroecondata/GDP.csv')
    getPredictionGDP(df)
def getData(values):
    for x in values:
        data = fred.get_series_latest_release(x)
        data = data.reset_index()
        data.columns = ['Date',  values[x]]
        data.to_csv(path + '/macroecondata/' + values[x] + '.csv')
#categories = ['UNrate', 'GDP', 'SP500','CPALTT01USM657N' , FPCPITOTLZGUSA (inflation), GDPC1 (real gdp), IMPGSC1 (real imports of goods), NETEXC (real exports), DGDSRX1Q020SBEA (personal consumption of goods), CORESTICKM159SFRBATL (CPI), DFF (Fed rate hikes), UMCSENT (consumer sentiment), RSXFS (retail sales), PPIACO (Producer price index), RECPROUSM156N (smoothed US rececssion probability)]


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

if __name__ == "__main__":
    main()