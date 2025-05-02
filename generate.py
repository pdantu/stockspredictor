import os
import json
import pandas as pd
from datetime import datetime
from yahooquery import Ticker
import time


path = os.getcwd()
sectorETF = ['XLK','XLF','XLU','XLI','XLE','XLV','XLP','XLY','XLC','XLRE','XLB']

def preprocessAll(etflist):
    stocks = {}
    for etf in etflist:
        df = pd.read_csv(f'{path}/holdings/{etf}-holdings.csv')
        df = df[~df['Symbol'].isin(['SSIXX', 'Other', 'NLOK'])]
        stocks[etf] = df['Symbol'].tolist()

    with open("stocks.json", "w") as outfile:
        json.dump(stocks, outfile, indent=4)
    return stocks

def getSectData(sector, stock_list):
    tickers = Ticker(stock_list, asynchronous=True)
    info_data = tickers.all_modules

    output = []
    today = datetime.now().strftime('%Y-%m-%d')

    for symbol in stock_list:
        data = info_data.get(symbol, {})
        summary = data.get('summaryDetail', {})
        key_stats = data.get('defaultKeyStatistics', {})
        financial = data.get('financialData', {})

        output.append({
            'ETF': sector,
            'Ticker': symbol,
            'Beta': key_stats.get('beta'),
            'Dividend_Yield': summary.get('dividendYield'),
            'Forward_PE': summary.get('forwardPE'),
            'Trailing_PE': summary.get('trailingPE'),
            'Market_Cap': summary.get('marketCap'),
            'Trailing_EPS': key_stats.get('trailingEps'),
            'Forward_EPS': key_stats.get('forwardEps'),
            'PEG_Ratio': key_stats.get('pegRatio'),
            'Price_To_Book': summary.get('priceToBook'),
            'EV_to_EBITDA': financial.get('enterpriseToEbitda'),
            'Free_Cash_Flow': financial.get('freeCashflow'),
            'Debt_to_Equity': financial.get('debtToEquity'),
            'Earnings_Growth': financial.get('earningsGrowth'),
            'Ebitda_Margins': financial.get('ebitdaMargins'),
            'Quick_Ratio': financial.get('quickRatio'),
            'Target_Mean_Price': financial.get('targetMeanPrice'),
            'Return_on_Equity': financial.get('returnOnEquity'),
            'Revenue_Growth': financial.get('revenueGrowth'),
            'Current_Ratio': financial.get('currentRatio'),
            'Current_Price': financial.get('currentPrice'),
            'date': today
        })

    df = pd.DataFrame(output)
    os.makedirs(f'{path}/metrics', exist_ok=True)
    df.to_csv(f'{path}/metrics/{sector}-metrics.csv', index=False)
    print(f"Saved: metrics/{sector}-metrics.csv")

def generateAll(stocks):
    for sector in stocks:
        print(f"Starting Sector: {sector}")
        getSectData(sector, stocks[sector])
        print(f"Finished Sector: {sector}")
        time.sleep(60)

def runAll():
    stocks = preprocessAll(sectorETF)
    print('finished preprocessing')
    generateAll(stocks)

def main():
    runAll()

if __name__ == "__main__":
    main()
