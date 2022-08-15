import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import math
path = os.getcwd()
print(path)
path = path[0:path.find("/extraPyFiles")]

spy = pd.read_csv(path + '/holdings/SPY-holdings.csv')


sectors = ['XLE', 'XLK', 'XLC', 'XLB', 'XLY', 'XLU', 'XLI', 'XLP', 'XLRE', 'XLF', 'XLV']

def rsi(df, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    #print(type(rsi))
    return rsi

def getScore(etf, stock, sharpe, columns):
    metricdf = pd.read_csv(path + '/metrics/' + etf + '-metrics.csv')
    metricdf = metricdf.fillna(0)
    factordict = {'Beta': -1 ,'Dividend Yield': 1, 'Forward P/E' : -1,'Trailing P/E': -1, 'Market Cap': 1, 'Trailing EPS': 1, 'Forward EPS': 1, 'PEG Ratio': -1, 'Price To Book': -1, 'E/V to EBITDA': -1, 'Free Cash Flow': 1, 'Deb To Equity': -1 ,'Earnings Growth': 1,'Ebitda margins': 1,'Quick Ratio': 1,'Target Mean Price': 1,'Return on Equity': 1 ,'Revenue Growth': 1,'Current Ratio': 1,'Current Price': 1}
    #print(metricdf.columns)
    subdf = metricdf[columns]
    #print(metricdf['Unnamed: 0'].head())
    score = 0
    metricdf.rename(columns={metricdf.columns[0]:"Symbol"}, inplace=True)
    #print(metricdf['Symbol'].head())
    tickerrow = metricdf[metricdf['Symbol'] == stock]
    for x in columns.keys():

        mean = metricdf[x].mean()
        sd = metricdf[x].std()
        val = tickerrow[x].iloc[0]
        if val - mean > 0:
            val = val - mean
            val = val / sd
            val = val * factordict.get(x) * columns.get(x)
        else:
            val = val - mean
            val = val / sd
            val = val * factordict.get(x) * columns.get(x)
        a = val / mean
        score += a
    score += sharpe * 10
    return(score)

def getETFaction(etf):
    a = yf.Ticker(etf)
    prices = a.history('max')
    prices['20DayEMA'] = prices['Close'].ewm(span = 20).mean()
    prices['100DayEMA'] = prices['Close'].ewm(span = 100).mean()
    if prices['20DayEMA'].iloc[len(prices) - 1] > prices['100DayEMA'].iloc[len(prices) - 1]:
        return(3)
    else:
        return(1)

#stocksdf = pd.DataFrame(columns=['Ticker', 'Technical Action', 'Score'])
portfolio = pd.DataFrame()
for x in sectors:
    stocksdf = pd.DataFrame(columns=['ETF', 'Ticker', 'Technical Action', 'Score'])
    df = pd.read_csv(path + '/holdings/' + x + '-holdings.csv')

    num = getETFaction(x)
    
    for y in df['Symbol']:
        if y == 'Other':
            continue
        if y == 'SSIXX':
            continue
        a  = yf.Ticker(y)
        
        prices = a.history('max')
        prices = prices.reset_index()
        if prices.shape[0] == 0:
            continue
        if prices.empty:
            continue
        prices['20DayEMA'] = prices['Close'].ewm(span = 20).mean()
        prices['100DayEMA'] = prices['Close'].ewm(span = 100).mean()
        prices['50DaySMA'] = prices['Close'].rolling(50).mean()
        prices['200DaySMA'] = prices['Close'].rolling(200).mean()
        prices['26DayEMA'] = prices['Close'].ewm(26).mean()
        prices['12DayEMA'] = prices['Close'].ewm(12).mean()
        prices['MACD'] = prices['12DayEMA'] - prices['26DayEMA']
        prices['MACD Signal'] = prices['MACD'].ewm(9).mean()
        subprices = prices.tail(253)
        z = subprices['Close'].diff()
        #print(y)
        #print(len(z))
        returns = []
        for i in range(1,len(z)):
            d = z.iloc[i] / subprices['Close'].iloc[i-1]
            returns.append(d)
        
        mean = sum(returns) / len(returns)
        variance = sum([((k - mean) ** 2) for k in returns]) / len(returns)
        res = variance ** 0.5
        sharpe = mean / res * math.sqrt(253)
        
        z = 0
        
        if prices['20DayEMA'].iloc[len(prices) - 1] > prices['100DayEMA'].iloc[len(prices) - 1]:
            z += 1
        else:
            z -= 1
        
        if prices['MACD'].iloc[len(prices) - 1] > prices['MACD Signal'].iloc[len(prices) - 1]:
            z += 1
        else:
            z -= 1
        prices['RSI'] = rsi(prices)
  
        for k in prices['RSI'].tail(10):
            if k > 70:
                z -= 1
        
        sc = getScore(x, y, sharpe, {'Forward EPS': 10, 'Forward P/E': 9, 'PEG Ratio': 8, 'Market Cap': 7, 'Price To Book': 6, 'Return on Equity': 5, 'Free Cash Flow': 4, 'Revenue Growth': 3, 'Dividend Yield': 2, 'Deb To Equity': 1})
        
        if z == 2:
            stocksdf.loc[len(stocksdf.index)] = [x, y, 'Buy', sc]
        else: 
            stocksdf.loc[len(stocksdf.index)] = [x, y, 'Sell', sc]

        
    stocksdf = stocksdf.sort_values(by=['Score'], ascending=False)
    buys = stocksdf[stocksdf['Technical Action'] == 'Buy']
    #buys = buys.head(num)
    strongbuys = buys[buys['Score'] > 0]
    portfolio = pd.concat([portfolio, strongbuys.head(num)])

    stocksdf.to_csv(path + '/results/' + x + '-action.csv')
    buys.to_csv(path + '/results/' + x + '-buys.csv')

scoresum = portfolio['Score'].sum()
portfolio['weight'] = (portfolio['Score'] / scoresum) * 100
portfolio['Dollar Amount'] = portfolio['weight'] / 100 * 5000
portfolio.to_csv(path + '/results/mainportfolio.csv')  
    



#newdf = pd.DataFrame(columns=['Ticker', 'Action'])
#pricesdf = pd.DataFrame(columns = ['Ticker', 'Profit', 'Hold Change'])


"""for i in range(1, len(row)):
        b = row[i]/xle.mean()[i-1]
        
        if i == 2 or i == 3 or i == 4 or i == 6 or i == 8:
            b = b - 1
            b = 10 * b
        elif i == 5:
            b = 5 * b
        elif i == len(row) - 1:
            b = 2 * b
        elif i == len(row) - 2:
            b = 12 * b
        else:
            b = 10 * b
        a += b
    
score.append(a)"""

#ssus = ssu.sort_values(by=['score'], ascending = False)


"""import nltk
from newspaper import Article
from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup as BS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import schedule
import time
    
def percentage(part, whole):
    return 100 * float(part)/float(whole)

def findSentiment(ticker):
    a = yf.Ticker(ticker)
    try:
        x = len(a.news)
    except:
        return(-1)
    #nltk.download('vader_lexicon')
    news = {}
    #print(len(a.news))
    if len(a.news) == 0:
        return 0
    for i in range(0, len(a.news)):
        url = a.news[i]['link']
        art = Article(url, language = 'en')
        art.download()
        art.parse()
        news[i+1] = art.text
        
    


    positive = 0
    negative = 0
    neutral = 0
    
    news_list = []
    pos_list = []
    neg_list = []
    neu_list = []
    
    newsdf = pd.DataFrame(news.items())
    newsdf.columns = ['Id', 'Text']
    #return(newsdf.shape)
    for news in newsdf['Text']:
        news_list.append(news)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
        neg = analyzer['neg']
        neu = analyzer['neu']
        pos = analyzer['pos']

        if neg > pos:
            neg_list.append(news)
            negative += 1
        elif pos > neg:
            pos_list.append(news)
            positive += 1
        elif pos == neg:
            neu_list.append(news)
            neutral += 1
    #return(positive)
    positive = percentage(positive, len(newsdf))
    negative = percentage(negative, len(newsdf))
    neutral = percentage(neutral, len(newsdf))
    #return(positive)
    if positive > negative:
        return(positive)
    elif positive < negative:
        return(0 - negative)
    else:
        return(0)
    
decision = []
for index, row in newdf.iterrows():
    #print(row['Ticker'])
    #print(decision)
    #print(len(decision))
    if row['Action'] == 'Buy':
        
        x = findSentiment(row['Ticker'])
        
        if x is None:
            decision.append('No Sentimetn')
        else:
            if x >= 87.5:
                decision.append('Strong Buy')
            elif x >= 67.5:
                decision.append('Buy')
            elif x == 0: 
                decision.append('Neutral')
            else:
                decision.append('Avoid')
    else:
        decision.append('Excluded')
newdf['Decision'] = decision
pe = []
eps = []
both = []
for index, row in newdf.iterrows():
    c = 0
    if row['Decision'] == 'Strong Buy':
        subs = xledata[xledata['Ticker'] == row['Ticker']]
        #print(type(subs['PE'].))
        if subs['PE'].iloc[0] < averagePe:
            c += 1
            pe.append('Below')
        else:
            pe.append('Above')
        if subs['Eps'].iloc[0] > averageEps:
            c += 1
            eps.append('Above')
        else:
            eps.append('Below')
        if c == 2:
            both.append('Yes')
        else:
            both.append('No')
    else:
        pe.append('Ignored')
        eps.append('Ignored')
        both.append('Ignored')
newdf['PE'] = pe
newdf['Eps'] = eps
newdf['Both'] = both"""
