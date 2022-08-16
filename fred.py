from fredapi import Fred
import pandas as pd
import yfinance as yf
import os

path = os.getcwd()

fred = Fred(api_key='13f1e0b5cbdcb8307bbf7bbca9852e4a')

def main():
    getData(['GDP', 'UNrate'])

def getData(values):
    for x in values:
        data = fred.get_series_latest_release(x)
        data = data.reset_index()
        data.columns = ['Date',  x]
        data.to_csv(path + '/macroecondata/' + x + '.csv')
#categories = ['UNrate', 'GDP', 'SP500','CPALTT01USM657N' , FPCPITOTLZGUSA (inflation), GDPC1 (real gdp), IMPGSC1 (real imports of goods), NETEXC (real exports), DGDSRX1Q020SBEA (personal consumption of goods), CORESTICKM159SFRBATL (CPI), DFF (Fed rate hikes), UMCSENT (consumer sentiment)]

if __name__ == "__main__":
    main()