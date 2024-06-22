import pandas as pd
from googlefinance import getQuotes
import yfinance as yf
df = pd.read_csv('merged.csv')
print(df[df['Ticker'] == 'AAPL'])
