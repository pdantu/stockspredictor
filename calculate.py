from cgitb import strong
from distutils.text_file import TextFile
from operator import attrgetter
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from fileinput import filename
import pandas as pd 
import numpy as np
import os
from os import listdir
import yfinance as yf
import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import math
path = os.getcwd()

def main(): 
    f_list = loop(path)  
    #calcResults(path,f_list)
    sendEmail(path)
    '''df = pd.read_csv('{0}/results/portfolio.csv'.format(path))
    df.sort_values(by='weight',inplace=True,ascending=False)
    df.to_csv('{0}/results/portfolio.csv'.format(path))'''


def sendEmail(path):
    sender_address = 'StocksPredictor123@outlook.com'
    sender_pass = 'Steelers2022!'
    #fileToSend = '{0}/results/portfolio.csv'.format(path)
    receiver_addresses = ['pdantu1234@gmail.com','archisdhar@gmail.com']
    attachments = ['{0}/results/portfolio.csv'.format(path),'{0}/results/sector_weights.csv'.format(path)]
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = 'list@stockspredictor'
    message['Subject'] = 'CSV attachment of portfolio'
    #The subject line
    #The body and the attachments for the mail
    mail_content = 'hey'
    
    for fileToSend in attachments:
        if '-' in fileToSend:
            name = fileToSend[fileToSend.find('results/') + 8:fileToSend.find('-')]
            name += '.csv'
        else:
            name = fileToSend[fileToSend.find('results/') + 8:]
        ctype, encoding = mimetypes.guess_type(fileToSend)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"

        maintype, subtype = ctype.split("/", 1)

        if maintype == "text":
            fp = open(fileToSend)
            # Note: we should handle calculating the charset
            attachment = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "image":
            fp = open(fileToSend, "rb")
            attachment = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == "audio":
            fp = open(fileToSend, "rb")
            attachment = MIMEAudio(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(fileToSend, "rb")
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(fp.read())
            fp.close()
            encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename=name)
        message.attach(attachment)

    server = smtplib.SMTP("smtp-mail.outlook.com",587)
    server.starttls()
    server.login(sender_address,sender_pass)
    server.sendmail(sender_address, receiver_addresses, message.as_string())
    server.quit()
    print('Mail Sent')




def calcResults(path,f_list):
    d_list = []
    for name in f_list:
        df = pd.read_csv('{0}/metrics/{1}'.format(path,name))
        print('Processing: ', name)
        d_list = process(d_list,df,name)

    portfolio = pd.concat(d_list)
    scoresum = portfolio['Score'].sum()
    portfolio['weight'] = (portfolio['Score'] / scoresum) * 100
    portfolio['Dollar Amount'] = portfolio['weight'] / 100 * 5000
    portfolio.sort_values(by='weight')
    portfolio.to_csv('{0}/results/portfolio.csv'.format(path))

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def loop(path):
    path += "/metrics"
    filenames = find_csv_filenames(path)
    return filenames


def process(d_list,df,sector):
    sector = sector[:sector.find("-")]
    #print(df.head())
    etfFraction = getETFaction(sector) #get the etf fraction of the sector
    stocksdf = pd.DataFrame(columns=['ETF', 'Ticker', 'Technical Action', 'Score'])
    for symbol in df['Unnamed: 0']:
        if symbol == 'Other':
            continue

        prices = getPrices(symbol)
        if prices.shape[0] == 0: continue
        subprices = prices.tail(253)
        z = subprices['Close'].diff()
       

        returns = []
        for i in range(1,len(z)):
            d = z.iloc[i] / subprices['Close'].iloc[i-1]
            returns.append(d)
        
        if returns:
            mean = sum(returns) / len(returns)
            variance = sum([((k - mean) ** 2) for k in returns]) / len(returns)
            res = variance ** 0.5
            sharpe = mean / res * math.sqrt(253)

        measure = 0
        
        if prices['20DayEMA'].iloc[len(prices) - 1] > prices['100DayEMA'].iloc[len(prices) - 1]:
            measure += 1
        else:
            measure -= 1
        
        if prices['MACD'].iloc[len(prices) - 1] > prices['MACD Signal'].iloc[len(prices) - 1]:
            measure += 1
        else:
            measure -= 1
        
        prices['RSI'] = rsi(prices)
  
        for k in prices['RSI'].tail(10):
            if k > 70:
                measure -= 1
        
        sc = getScore(sector, symbol, sharpe, {'Forward EPS': 10, 'Forward P/E': 9, 'PEG Ratio': 8, 'Market Cap': 7, 'Price To Book': 6, 'Return on Equity': 5, 'Free Cash Flow': 4, 'Revenue Growth': 3, 'Dividend Yield': 2, 'Deb To Equity': 1})
        if measure == 2:
            stocksdf.loc[len(stocksdf.index)] = [sector, symbol, 'Buy', sc]
        else: 
            stocksdf.loc[len(stocksdf.index)] = [sector, symbol, 'Sell', sc]

    stocksdf = stocksdf.sort_values(by=['Score'], ascending=False)
    buys = stocksdf[stocksdf['Technical Action'] == 'Buy']
    stocksdf.to_csv(path + '/results/' + sector + '-action.csv')
    buys.to_csv(path + '/results/' + sector + '-buys.csv')
    if (sector == 'QQQ' or sector =='SPY'):
        return d_list
    strongbuys = buys[buys['Score'] > 0]
    x = buys[buys['Score'] > 70]
    if x.shape[0] > 3:
        etfFraction = x.shape[0]
    d_list.append(strongbuys.head(etfFraction))

    return d_list



def getPrices(ticker):
    a  = yf.Ticker(ticker)  
    prices = a.history('max')
    prices = prices.reset_index()
    if prices.shape[0] == 0:
        return prices
    prices['20DayEMA'] = prices['Close'].ewm(span = 20).mean()
    prices['100DayEMA'] = prices['Close'].ewm(span = 100).mean()
    prices['50DaySMA'] = prices['Close'].rolling(50).mean()
    prices['200DaySMA'] = prices['Close'].rolling(200).mean()
    prices['26DayEMA'] = prices['Close'].ewm(26).mean()
    prices['12DayEMA'] = prices['Close'].ewm(12).mean()
    prices['MACD'] = prices['12DayEMA'] - prices['26DayEMA']
    prices['MACD Signal'] = prices['MACD'].ewm(9).mean()
    return prices

def getETFaction(etf):
    prices = getPrices(etf)
    if prices['20DayEMA'].iloc[len(prices) - 1] > prices['100DayEMA'].iloc[len(prices) - 1]:
        return(3)
    else:
        return(1)

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
    
    
def createGraphic(path):
    portfolio = pd.read_csv('{0}/results/portfolio.csv'.format(path))
    filenames = loop(path,False)
    print(filenames)
    df2 = portfolio.groupby(['ETF'])['weight'].sum().reset_index()
    
    df2.sort_values(by='weight')
    df2.to_csv('{0}/results/sector_weights.csv'.format(path))


if __name__ == "__main__":
    main()