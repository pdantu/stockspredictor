from asyncio.base_subprocess import ReadSubprocessPipeProto
import pandas as pd
import numpy as np
import json
import csv
import os
import yfinance as yf
from yahooquery import Ticker
from yahooquery.utils import _convert_to_timestamp, _flatten_list, _history_dataframe
import sqlite3
from datetime import datetime
path = os.getcwd()
sectorETF = ['XLK','XLF','XLU','XLI','XLE','XLV','XLP','XLY','XLC','XLRE','XLB','XLC']

import sqlite3

def create_table():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY,
            ETF TEXT,
            Ticker TEXT,
            Beta REAL,
            Dividend_Yield REAL,
            Forward_PE REAL,
            Trailing_PE REAL,
            Market_Cap REAL,
            Trailing_EPS REAL,
            Forward_EPS REAL,
            PEG_Ratio REAL,
            Price_To_Book REAL,
            EV_to_EBITDA REAL,
            Free_Cash_Flow REAL,
            Debt_to_Equity REAL,
            Earnings_Growth REAL,
            Ebitda_Margins REAL,
            Quick_Ratio REAL,
            Target_Mean_Price REAL,
            Return_on_Equity REAL,
            Revenue_Growth REAL,
            Current_Ratio REAL,
            Current_Price REAL,
            date TEXT
        )
    ''')

    conn.commit()
    conn.close()


def main():
    # print(1)
    drop_table()
    print(path)
    create_table()
    # singleStoc$kData('AAPL')
    runAll()
    
def drop_table():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS metrics')
    conn.commit()
    conn.close()

def runAll():
    stocks = preprocessAll(sectorETF)
    print('finished preprocesing')
    f = open('stocks.json','r')
    stocks = json.load(f)
    generateAll(stocks)

def runOne(sector):
    f = open('{0}/stocks.json'.format(path),'r')
    stocks = json.load(f)
    getSectData(sector,stocks[sector])
    f.close()

def addOne(sector):
    res = preprocessOne(sector)
    getSectData(sector,res[sector])

def generateAll(stocks):
    for sector in stocks.keys():
        print("Starting Sector: ", sector)
        getSectData(sector,stocks[sector])
        print("Finished Sector: ", sector)

def preprocessAll(etflist):   ## modify this to read from csv and get the hodlings that way. 
    stocks = {}
    for etf in etflist:
        df = pd.read_csv('{0}/holdings/{1}-holdings.csv'.format(path,etf))
        df = df[(df['Symbol'] != 'SSIXX') & (df['Symbol'] != 'Other') & (df['Symbol'] != 'NLOK')]
        x = df['Symbol']
        symbols = x.to_numpy()
        symbols = list(symbols)
        stocks[etf] = symbols
        
    
    with open("stocks.json", "w") as outfile:
        json.dump(stocks, outfile,indent=4)
    return stocks

def preprocessOne(etf):
    stocks = {}
    df = pd.read_csv('{0}/holdings/{1}-holdings.csv'.format(path,etf))
    df = df[(df['Symbol'] != 'SSIXX') & (df['Symbol'] != 'Other')]
    x = df['Symbol']
    symbols = x.to_numpy()
    symbols = list(symbols)
    stocks[etf] = symbols

    with open("stocks.json", "a") as outfile:
        json.dump(stocks, outfile,indent=4)
    return stocks


def getSectData(sector,stock_list): #singlesectorDict
    newDict = {}
    newDict[sector] = {}
    for stock in stock_list: 
        newDict[sector][stock] = {}
        newDict[sector][stock] = singleStockData(stock)
    
    df = pd.DataFrame.from_dict(newDict[sector],orient='index')
    df = df.reset_index()
    df = df.rename(columns={'index': 'Ticker'})
    # df.to_csv('{0}/metrics/{1}-metrics.csv'.format(path,sector))
    df = df.rename(columns={
        'Dividend Yield': 'Dividend_Yield',
        'Forward P/E': 'Forward_PE',
        'Trailing P/E': 'Trailing_PE',
        'Market Cap': 'Market_Cap',
        'Trailing EPS': 'Trailing_EPS',
        'Forward EPS': 'Forward_EPS',
        'PEG Ratio': 'PEG_Ratio',
        'Price To Book': 'Price_To_Book',
        'E/V to EBITDA': 'EV_to_EBITDA',
        'Free Cash Flow': 'Free_Cash_Flow',
        'Debt to Equity': 'Debt_to_Equity',
        'Earnings Growth': 'Earnings_Growth',
        'Ebitda Margins': 'Ebitda_Margins',
        'Quick Ratio': 'Quick_Ratio',
        'Target Mean Price': 'Target_Mean_Price',
        'Return on Equity': 'Return_on_Equity',
        'Revenue Growth': 'Revenue_Growth',
        'Current Ratio': 'Current_Ratio',
        'Current Price': 'Current_Price'
    })

    df['ETF'] = sector
    df['date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Connect to SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect('data.db')
    
    # Save the DataFrame to the SQLite database
    df.to_sql('metrics', conn, if_exists='append', index=False)
    
    conn.close()
    

def singleStockData(stock):
    answer = {}
    ticker = yf.Ticker(stock)
    
    try:
        info = ticker.info
        if info:
            try:
                answer['Beta'] = info.get('beta', None)
                answer['Dividend Yield'] = info.get('dividendYield', None)
                answer['Forward P/E'] = info.get('forwardPE', None)
                answer['Trailing P/E'] = info.get('trailingPE', None)
                answer['Market Cap'] = info.get('marketCap', None)
                answer['Trailing EPS'] = info.get('trailingEps', None)
                answer['Forward EPS'] = info.get('forwardEps', None)
                answer['PEG Ratio'] = info.get('pegRatio', None)
                answer['Price To Book'] = info.get('priceToBook', None)
                answer['E/V to EBITDA'] = info.get('enterpriseToEbitda', None)
                answer['Free Cash Flow'] = info.get('freeCashflow', None)
                answer['Debt to Equity'] = info.get('debtToEquity', None)
                answer['Earnings Growth'] = info.get('earningsQuarterlyGrowth', None)
                answer['Ebitda Margins'] = info.get('ebitdaMargins', None)
                answer['Quick Ratio'] = info.get('quickRatio', None)
                answer['Target Mean Price'] = info.get('targetMeanPrice', None)
                answer['Return on Equity'] = info.get('returnOnEquity', None)
                answer['Revenue Growth'] = info.get('revenueGrowth', None)
                answer['Current Ratio'] = info.get('currentRatio', None)
                answer['Current Price'] = info.get('currentPrice', None)
            except Exception as e:
                print(stock)
    except Exception as e:
        print(stock)

    return answer
    # answer = {}
    # #a = Ticker('AAPL', asynchronous=True)
    # ticker = Ticker(stock, asynchronous=True)
    # print(ticker.summary_detail)
    # try:
    #     summary = ticker.summary_detail[stock]
    #     print(summary)
    #     # print(summary)
    #     default = ticker.key_stats[stock]
    #     finance = ticker.financial_data[stock]
    
    # #print(default)
    #     if summary: # use summary_detail method
    #         try:
    #             beta = summary.get('beta')
    #             if type(beta) == dict:
    #                 answer['Beta'] = None
    #             else:
    #                 answer['Beta'] = beta   # good

    #             divY = summary.get('dividendYield')
    #             if type(divY) == dict:
    #                 answer['Dividend Yield'] = None
    #             else:
    #                 answer['Dividend Yield'] = divY  # good

    #             forwardPE = summary.get('forwardPE')
    #             if type(forwardPE) == dict:
    #                 answer['Forward P/E'] = None
    #             else:
    #                 answer['Forward P/E'] = forwardPE # good

    #             trailingPE = summary.get('trailingPE')
    #             if type(trailingPE) == dict:
    #                 answer['Trailing P/E'] = None
    #             else:
    #                 answer['Trailing P/E'] = trailingPE # good

    #             marketCap = summary.get('marketCap')
    #             if type(marketCap) == dict:
    #                 answer['Market Cap'] = None
    #             else:
    #                 answer['Market Cap'] = marketCap # 
    #         except Exception as e: 
    #             print(stock)

        
    #     if default: # use key_stats method
    #         try:
    #             trailingEPS = default.get('trailingEps')
    #             if type(trailingEPS) == dict:
    #                 answer['Trailing EPS'] = None
    #             else:
    #                 answer['Trailing EPS'] = trailingEPS 

    #             forwardEPS = default.get('forwardEps')
    #             if type(forwardEPS) == dict:
    #                 answer['Forward EPS'] = None
    #             else:
    #                 answer['Forward EPS'] = forwardEPS

    #             pegRatio = default.get('pegRatio')
    #             if type(pegRatio) == dict:
    #                 answer['PEG Ratio'] = None
    #             else:
    #                 answer['PEG Ratio'] = pegRatio

    #             priceToBook = default.get('priceToBook')
    #             if type(priceToBook) == dict:
    #                 answer['Price To Book'] = None
    #             else:
    #                 answer['Price To Book'] = priceToBook

    #             evtoeb = default.get('enterpriseToEbitda')
    #             if type(evtoeb) == dict:
    #                 answer['E/V to EBITDA'] = None
    #             else:
    #                 answer['E/V to EBITDA'] = evtoeb
    #         except Exception as e: 
    #             print(stock)

        
    #     if finance:  #get from financial_data
    #         try:
    #             freeCashFlow = finance.get('freeCashflow')
    #             if type(freeCashFlow) == dict:
    #                 answer['Free Cash Flow'] = None
    #             else:
    #                 answer['Free Cash Flow'] = freeCashFlow

    #             debtToEquity = finance.get('debtToEquity')
    #             if type(debtToEquity) == dict:
    #                 answer['Deb to Equity'] = None
    #             else:
    #                 answer['Deb To Equity'] = debtToEquity

    #             earningsGrowth = finance.get('earningsGrowth')
    #             if type(earningsGrowth) == dict:
    #                 answer['Earnings Growth'] = None
    #             else:
    #                 answer["Earnings Growth"] = earningsGrowth

    #             ebitdaMargins = finance.get('ebitdaMargins')
    #             if type(ebitdaMargins) == dict:
    #                 answer['Ebitda margins'] = None
    #             else:
    #                 answer['Ebitda margins'] = ebitdaMargins

    #             quickRatio = finance.get('quickRatio')
    #             if type(quickRatio) == dict:
    #                 answer['Quick Ratio'] = None
    #             else:
    #                 answer['Quick Ratio'] = quickRatio

    #             targetMeanPrice = finance.get('targetMeanPrice')
    #             if type(targetMeanPrice) == dict:
    #                 answer['Target Mean Price'] = None
    #             else:
    #                 answer['Target Mean Price'] = targetMeanPrice

    #             returnOnEquity = finance.get('returnOnEquity')
    #             if type(returnOnEquity) == dict:
    #                 answer['Return on Equity'] = None
    #             else:
    #                 answer['Return on Equity'] = returnOnEquity

    #             revenueGrowth = finance.get('revenueGrowth')
    #             if type(revenueGrowth) == dict:
    #                 answer['Revenue Growth'] = None
    #             else:
    #                 answer['Revenue Growth'] = revenueGrowth

    #             currentRatio = finance.get('currentRatio')
    #             if type(currentRatio) == dict:
    #                 answer['Current Ratio'] = None
    #             else:
    #                 answer["Current Ratio"] = currentRatio

    #             currentPrice = finance.get('currentPrice')
    #             if type(currentPrice) == dict:
    #                 answer['Current Price'] = None
    #             else:
    #                 answer['Current Price'] = currentPrice
    #         except:
    #             print(stock)
    # except Exception as e:
    #     print(stock)

    # return answer



if __name__ == "__main__":
    main()

