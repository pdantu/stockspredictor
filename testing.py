import pandas as pd
from googlefinance import getQuotes
import yfinance as yf
df = pd.read_csv('merged.csv')
tickers = pd.read_csv('tickers.csv')

a = df[df['Ticker'] == 'AAPL']
lis = list(df['Ticker'])
for x in tickers['Ticker']:
    if x == 'AAPL':
        continue
    if x in lis:
        sub = df[df['Ticker'] == x]
        a = pd.concat([a, sub], ignore_index=True)
a = a[['ETF', 'Ticker', 'Technical Action', 'Score']]
a.to_csv('stocks.csv')