import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
path = os.getcwd()
print(path)

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

stocks = []
for x in sectors:
    df = pd.read_csv('/Users/archisdhar/Downloads/' + x + '-holdings.csv')
    
    for y in df['Symbol']:
        print(y)
        a  = yf.Ticker(y)
        prices = a.history('max')
        prices = prices.reset_index()
        if prices.shape[0] == 0:
            continue
        prices['50DayEMA'] = prices['Close'].ewm(span = 50).mean()
        prices['200DayEMA'] = prices['Close'].ewm(span = 200).mean()
        prices['50DaySMA'] = prices['Close'].rolling(50).mean()
        prices['200DaySMA'] = prices['Close'].rolling(200).mean()
        prices['26DayEMA'] = prices['Close'].ewm(26).mean()
        prices['12DayEMA'] = prices['Close'].ewm(12).mean()
        prices['MACD'] = prices['12DayEMA'] - prices['26DayEMA']
        prices['MACD Signal'] = prices['MACD'].ewm(9).mean()
        subprices = prices.tail(253)
        
        z = 0
        
        if prices['50DayEMA'].iloc[len(prices) - 1] > prices['200DayEMA'].iloc[len(prices) - 1]:
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
        
        if z == 2:
            stocks.append(y)
        
        
        
print(len(stocks))
print(stocks)    
        
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import os

xlk = pd.read_csv('/Users/archis/Downloads/XLK-holdings.csv', index_col=[0])


c = yf.Ticker('ABBV')
cdf = c.history('max')
cdf = cdf.reset_index()
cdf['20DaySMA'] = cdf['Close'].ewm(span = 50).mean()
cdf['100DaySMA'] = cdf['Close'].ewm(span = 200).mean()
cdf['12DayEMA'] = cdf['Close'].ewm(span = 12).mean()
cdf['26DayEma'] = cdf['Close'].ewm(span = 26).mean()
cdf['MACD'] = cdf['12DayEMA'] - cdf['26DayEma']
cdf['MACDema'] = cdf['MACD'].ewm(span = 9).mean()

cdf = cdf.tail(230)
plt.plot(cdf['Date'], cdf['Close'], color = 'red')
plt.plot(cdf['Date'], cdf['20DaySMA'], color = 'blue')
plt.plot(cdf['Date'], cdf['100DaySMA'], color = 'orange')
plt.show()
plt.plot(cdf['Date'], cdf['MACD'], color = 'green')
plt.plot(cdf['Date'], cdf['MACDema'], color = 'black')
plt.show()

xlk = xlk.reset_index()

a = yf.Ticker('SPY')
da = a.history('max')
da = da.reset_index()



#a = yf.Ticker('HES')

#xledata = pd.read_csv('/Users/archisdhar/Stocks project/Data/xledata.csv')

#xledata.columns = ['Ticker', 'PE', 'priceToBook', 'Eps', 'returnOnEquity','freeCashflow', 'revenueGrowth']

#averagePe = xledata['PE'].mean()
#averageEps = xledata['Eps'].mean()
newdf = pd.DataFrame(columns=['Ticker', 'Action'])
pricesdf = pd.DataFrame(columns = ['Ticker', 'Profit', 'Hold Change'])

corrdict = {}

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
s = []
pricern = []
rsios = []
for x in xlk['Symbol']:
    #print(x)
    if x == 'SSIXX':
        continue
    """if x == 'Other':
        continue
    if x == 'BF.B':
        continue
    if x == 'ROP':
        continue
    if x == 'APH':
        continue
    if x == 'AMD':
        continue
    if x == 'SEDG:':
        continue"""
    a = yf.Ticker(x)
    df = a.history(period = 'max')
    if df.shape[0] == 0:
        continue
    df = df.reset_index()
    df = df[['Date', 'Close']]
    #print(df['Close'].corr(da['Close']))
    #corrdict[x] = df['Close'].corr(da['Close'])
    corrdf = df.merge(da, how='inner', on='Date')
    #print(corrdf.tail())
    #print(corrdf.columns)
    corrdict[x] = corrdf['Close_x'].corr(corrdf['Close_y'])
    #print(corrdict)
    df['20DaySMA'] = df['Close'].ewm(span = 50).mean()
    df['100DaySMA'] = df['Close'].ewm(span = 200).mean()
    df['EMA'] = df['Close'].ewm(span = 200).mean()
    df['RSI'] = rsi(df)
    s.append(df['RSI'].iloc[len(df) - 1])
    subdf = df.tail(140)
    i = 0
    price = 0
    profit = 0
    currentprice = subdf['Close'].iloc[len(subdf) -1]
    pricern.append(currentprice)
    pastprice = subdf['Close'].iloc[0]
    if subdf['20DaySMA'].iloc[0] < subdf['100DaySMA'].iloc[0]:
        count = 0
        for index, row in subdf.iterrows():
            if row['20DaySMA'] > row['100DaySMA']:
                i = 1
                if count == 0:
                    count += 1
                    price = row['Close']
            if i > 0 and row['20DaySMA'] < row['100DaySMA']:
                i = -1
                profit += row['Close'] - price
                
            if i < 0 and row['20DaySMA'] > row['100DaySMA']:
                i = 1
                price = row['Close']
    if subdf['20DaySMA'].iloc[0] > subdf['100DaySMA'].iloc[0]:
        c = 0
        price = subdf['Close'].iloc[0]
        for index, row in subdf.iterrows():
            #print(i)
            if row['20DaySMA'] < row['100DaySMA'] and c == 0:
                c += 1
                i = -1
            if i < 0 and row['20DaySMA'] > row['100DaySMA']:
                i = 1
                #print(price)
                price = row['Close']
                """"print(row['Date'])
                print(price)
                print(i)"""
            if i > 0 and row['20DaySMA'] < row['100DaySMA']:
                i = -1
                """print(row['Date'])
                print(row['Close'])"""
                profit += row['Close'] - price
                #print(profit)
    if i >= 0:
        profit += subdf['Close'].iloc[len(subdf) - 1] - price
        newdf.loc[len(newdf.index)] = [x, 'Buy']
    elif i < 0:
        newdf.loc[len(newdf.index)] = [x, 'Sell']
    #else:
        #newdf.loc[len(newdf.index)] = [x, 'Hold']
    #subdf.to_csv('aapl.csv')
    #print(profit)
    #print(newdf.loc(len(newdf.index) - 1))
    """plt.plot(subdf['Date'], subdf['Close'], color = 'red')
    plt.plot(subdf['Date'], subdf['20DaySMA'], color = 'blue')
    plt.plot(subdf['Date'], subdf['100DaySMA'], color = 'orange')
    plt.show()"""
    #print(profit)
    ar = 0
    for x in subdf['RSI'].tail(10):
        if x > 70:
            ar += 1
            rsios.append('Yes')
            break
    if ar == 0:
        rsios.append('No')
    pricesdf.loc[len(pricesdf.index)] = [x, profit, currentprice - pastprice]
    #break

newdf['RSI'] = s

newdf['price'] = pricern
newdf['Went Over 70'] = rsios

xle = pd.read_csv(path + '/metrics/XLE-metrics.csv')
xle = pd.read_csv('/Users/archis/xlk.csv')

#print(xle.mean()[0])

xle.rename(columns={'Unnamed: 0':'Ticker'}, inplace=True)
"""print(newdf.shape)
print(xle.shape)
print(xle['Ticker'].tolist())
print(newdf['Ticker'].tolist())"""
score = []  
c = 0  
b = 0
for index, row in xle.iterrows():
    a = 0
    if row['Ticker'] not in newdf['Ticker'].tolist():
        #print(row['Ticker'])
        b += 1
        continue
    
    c += 1
    
    
    for i in range(1, len(row)):
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
    
    score.append(a)
#print(b)
#print(c)    
newdf['score'] = score
   
#print(newdf.shape)
su = newdf[newdf['Action'] == 'Buy']
#print(su.shape)
ssu = su[su['RSI'] < 70]
ssu = ssu[ssu['Went Over 70'] == 'No']
#print(ssu.shape)
ssus = ssu.sort_values(by=['score'], ascending = False)
avg = ssus['score'].mean()
ssuss = ssus[ssus['score'] > avg]
#print(type(ssus))
#print(ssuss.shape)
print(ssuss.head(8))
#print(ssuss.head(8)['price'].sum())
#print(ssu['price'].sum())


sorted_dict = {}
sorted_keys = sorted(corrdict, key=corrdict.get)  # [1, 3, 2]

for w in sorted_keys:
    sorted_dict[w] = corrdict[w]

#print(sorted_dict)



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
newdf.to_csv('xleaction.csv')
#pricesdf.to_csv('xlevalue.csv')