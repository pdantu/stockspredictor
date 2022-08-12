from asyncio.base_subprocess import ReadSubprocessPipeProto
import pandas as pd
import numpy as np
import json
import yfinance as yf
import csv
import os
path = os.getcwd()
sectorETF = ['XLK','XLF','XLU','XLI','XLE','XLV','XLP','XLY','XLC','XLRE','XLB','XLC']

def main():
    print(path)
    print('hello')

def runAll():
    stocks = preprocessAll(sectorETF)
    print('finished preprocesing')
    f = open('stocks.json','r')
    stocks = json.load(f)
    generateAll(stocks)

def runOne(sector):
    f = open('stocks.json','r')
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
        df = df[(df['Symbol'] != 'SSIX') & (df['Symbol'] != 'Other')]
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
    df = df[(df['Symbol'] != 'SSIX') & (df['Symbol'] != 'Other')]
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

    with open("{0}/jsons/{1}.json".format(path,sector), "w") as outfile:
        json.dump(newDict[sector], outfile,indent=4)
    
    df = pd.DataFrame.from_dict(newDict[sector],orient='index')
    df.to_csv('{0}/metrics/{1}-metrics.csv'.format(path,sector))
    
    

def singleStockData(stock):
    answer = {}
    ticker = yf.Ticker(stock).stats()
    finance = ticker.get('financialData')
    summary = ticker.get('summaryDetail')
    default = ticker.get('defaultKeyStatistics')
     
    if summary:
        beta = summary.get('beta')
        answer['Beta'] = beta

        divY = summary.get('dividendYield')
        answer['Dividend Yield'] = divY

        forwardPE = summary.get('forwardPE')
        answer['Forward P/E'] = forwardPE

        trailingPE = summary.get('trailingPE')
        answer['Trailing P/E'] = trailingPE

        marketCap = summary.get('marketCap')
        answer['Market Cap'] = marketCap

    
    if default:
        trailingEPS = default.get('trailingEps')
        answer['Trailing EPS'] = trailingEPS

        forwardEPS = default.get('forwardEps')
        answer['Forward EPS'] = forwardEPS

        pegRatio = default.get('pegRatio')
        answer['PEG Ratio'] = pegRatio

        priceToBook = default.get('priceToBook')
        answer['Price To Book'] = priceToBook

        evtoeb = default.get('enterpriseToEbitda')
        answer['E/V to EBITDA'] = evtoeb

    
    if finance:
        freeCashFlow = finance.get('freeCashflow')
        answer['Free Cash Flow'] = freeCashFlow

        debtToEquity = finance.get('debtToEquity')
        answer['Deb To Equity'] = debtToEquity

        earningsGrowth = finance.get('earningsGrowth')
        answer["Earnings Growth"] = earningsGrowth

        ebitdaMargins = finance.get('ebitdaMargins')
        answer['Ebitda margins'] = ebitdaMargins

        quickRatio = finance.get('quickRatio')
        answer['Quick Ratio'] = quickRatio

        targetMeanPrice = finance.get('targetMeanPrice')
        answer['Target Mean Price'] = targetMeanPrice

        returnOnEquity = finance.get('returnOnEquity')
        answer['Return on Equity'] = returnOnEquity

        revenueGrowth = finance.get('revenueGrowth')
        answer['Revenue Growth'] = revenueGrowth

        currentRatio = finance.get('currentRatio')
        answer["Current Ratio"] = currentRatio

        currentPrice = finance.get('currentPrice')
        answer['Current Price'] = currentPrice


    return answer

if __name__ == "__main__":
    main()

