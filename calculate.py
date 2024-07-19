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
from datetime import date
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import sqlite3
from datetime import datetime

class CalculateStocks:
    def drop_table():
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS results')
        conn.commit()
        conn.close()
    
    def create_table():
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action (
                id INTEGER PRIMARY KEY,
                ETF TEXT,
                Ticker TEXT,
                Technical_Action TEXT,
                Score REAL,
                date TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def main(self): 
        # etfs = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # # etfs = ['XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
        # # etfs = ['XLK']
        # for x in etfs:
        #     self.calculateSentiments(x, tokenizer, model)

        f_list = self.loop(self.path,False)
        # types = ['growth', 'value', 'income']
        types = ['growth']
        for x in types:
            self.calcResults(self.path,f_list, x)
            df = pd.read_csv('{0}/portfolio/portfolio{1}.csv'.format(self.path, x))  
            self.writePortfolioToLogs(self.path,df)
       
        # df = pd.read_csv('{0}/portfolio/portfoliovalue.csv'.format(self.path))  
        # self.addCompName(df, 'value')
        # df = pd.read_csv('{0}/portfolio/portfolioincome.csv'.format(self.path))  
        # self.addCompName(df, 'income')
        df = pd.read_csv('{0}/portfolio/portfoliogrowth.csv'.format(self.path))  
        self.addCompName(df, 'growth')
        #getSentiment(f_list)
         # #findDifference('{0}/logs/2022-08-18_portfolio.csv'.format(path),'{0}/portfolio/portfolio.csv'.format(path))
        # self.sendEmail(self.path)
        # self.sendEmail(self.path)
        # self.stockPrediction('AAPL')
    def stockPrediction(self, stock):
        s = yf.Ticker(stock)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        import matplotlib.pyplot as plt

        # Assuming you have a pandas DataFrame called 'data' with a column named 'TimeSeriesData'
        # Ensure the 'TimeSeriesData' column is in the appropriate format (e.g., numeric)

        # Convert the time series data into a numpy array
        col = s.history('max')
        col = col['Close']
        col = col.dropna()
        data_array = col.values.reshape(-1, 1)

        # Perform data normalization using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_array)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

        # Define the number of previous time steps to consider for each prediction
        n_steps = 10

        # Create input sequences and corresponding target values
        def create_sequences(data, n_steps):
            X = []
            y = []
            for i in range(n_steps, len(data)):
                X.append(data[i - n_steps:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        train_X, train_y = create_sequences(train_data, n_steps)
        test_X, test_y = create_sequences(test_data, n_steps)

        # Reshape the input sequences to fit the LSTM input shape
        train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
        test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the LSTM model
        model.fit(train_X, train_y, epochs=10, batch_size=32)

        # Make predictions on the test data
        predictions = model.predict(test_X)

        # Inverse transform the scaled predictions and actual values to their original scale
        predictions = scaler.inverse_transform(predictions)
        actual_values = scaler.inverse_transform(test_y.reshape(-1, 1))
        print(test_X[-1])
        # Plot the predicted values and actual values
        plt.plot(predictions, label='Predicted')
        plt.plot(actual_values, label='Actual')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def addCompName(self, data, type):
        names = []
        for x in data['Ticker']:
            a = yf.Ticker(x)
            name = a.info['longName']
            names.append(name)
        data['Name'] = names
        data.to_csv('{0}/portfolio/portfolio{1}.csv'.format(self.path, type))

    def sendEmail(self, path):
        sender_address = 'StocksPredictor123@outlook.com'
        sender_pass = input("Email Pasword: ") #TODO -------------------------------------------------> ADD PASSWORD
        #fileToSend = '{0}/results/portfolio.csv'.format(path)
        receiver_addresses = ['pdantu1234@gmail.com', 'archisdhar@gmail.com', 'abhilash.gogineni@gmail.com', 'anish.t2023@gmail.com']
        # attachments = ['{0}/portfolio/portfolio.csv'.format(path),'{0}/portfolio/sector_weights.csv'.format(path),'{0}/portfolio/actions.csv'.format(path)]
        attachments = ['{0}/portfolio/portfoliogrowth.csv'.format(path),'{0}/portfolio/portfoliovalue.csv'.format(path),'{0}/portfolio/portfolioincome.csv'.format(path)]
        #Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = 'list@stockspredictor'
        message['Subject'] = 'Portfolio after {0} trading hours'.format(date.today())
        #The subject line
        #The body and the attachments for the mail
        mail_content = 'hey'
        
        for fileToSend in attachments:
            if '-' in fileToSend:
                name = fileToSend[fileToSend.find('portfolio/') + 10:fileToSend.find('-')]
                name += '.csv'
            else:
                name = fileToSend[fileToSend.find('portfolio/') + 10:]
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




    def calcResults(self, path,f_list,type):
        d_list = []
        for name in f_list:
            if 'SPY' in name or 'QQQ' in name:
                continue
            df = pd.read_csv('{0}/metrics/{1}'.format(path,name))
            print('Processing: ', name)
            d_list = self.process(d_list,df,name, type)

        portfolio = pd.concat(d_list)
        scoresum = portfolio['Score'].sum()
        portfolio['weight'] = (portfolio['Score'] / scoresum) * 100
        portfolio['Dollar Amount'] = portfolio['weight'] / 100 * 5000
        portfolio.sort_values(by='weight',inplace=True,ascending=False)
        portfolio.to_csv('{0}/portfolio/portfolio{1}.csv'.format(path, type))
        self.createGraphic(path)

    def find_csv_filenames(self, path_to_dir, suffix=".csv" ):
        filenames = listdir(path_to_dir)
        return [ filename for filename in filenames if filename.endswith( suffix ) ]

    def loop(self, path,results):
        if results:
            path += "results"
        else:
            path += "/metrics"
        filenames = self.find_csv_filenames(path)
        return filenames


    def process(self, d_list,df,sector, type):
        sector = sector[:sector.find("-")]
        #print(df.head())
        etfFraction = self.getETFaction(sector) #get the etf fraction of the sector
        stocksdf = pd.DataFrame(columns=['ETF', 'Ticker', 'Technical Action', 'Score'])
        for symbol in df['Unnamed: 0']:
            if symbol == 'Other':
                continue

            prices = self.getPrices(symbol)
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
            
            '''
            if prices['MACD'].iloc[len(prices) - 1] > prices['MACD Signal'].iloc[len(prices) - 1]:
                measure += 1
            else:
                measure -= 1
            '''
            
            
            prices['RSI'] = self.rsi(prices)
    
            for k in prices['RSI'].tail(10):
                if k > 100:
                    measure -= 1
            if type == 'growth':
                # {'Forward EPS': 3, 'Forward P/E': 3, 'PEG Ratio': 3, 'Market Cap': 1, 'Price To Book': 1, 'Return on Equity': 3, 'Free Cash Flow': 1, 'Revenue Growth': 3, 'Dividend Yield': 1, 'Debt To Equity': 1, 'Earnings Growth': 3}
                self.weightdict = {'Forward EPS': 3, 'Forward P/E': 3, 'PEG Ratio': 3, 'Market Cap': 1, 'Price To Book': 1, 'Return on Equity': 3, 'Free Cash Flow': 1, 'Revenue Growth': 3, 'Dividend Yield': 1, 'Debt to Equity': 1, 'Earnings Growth': 3}
            elif type == 'value':
                self.weightdict = {'Trailing P/E': 3, 'Forward P/E': 3, 'PEG Ratio': 1, 'Market Cap': 1, 'Price To Book': 3, 'Return on Equity': 1, 'Free Cash Flow': 3, 'Revenue Growth': 1, 'Dividend Yield': 3, 'Debt to Equity': 1, 'E/V to EBITDA': 3, 'Beta': 3}
            elif type == 'income':
                self.weightdict = {'Forward EPS': 3, 'Forward P/E': 1, 'PEG Ratio': 1, 'Market Cap': 1, 'Price To Book': 1, 'Return on Equity': 1, 'Free Cash Flow': 3, 'Revenue Growth': 1, 'Dividend Yield': 3, 'Debt to Equity': 3, 'Trailing EPS': 3}





            # {'Forward EPS': 10, 'Forward P/E': 9, 'PEG Ratio': 8, 'Market Cap': 7, 'Price To Book': 6, 'Return on Equity': 5, 'Free Cash Flow': 4, 'Revenue Growth': 3, 'Dividend Yield': 2, 'Deb To Equity': 1}
            
            sc = self.getScore(sector, symbol, sharpe, self.weightdict)
            # sentiment = self.getSentiment(symbol)
            if measure == 1:
                stocksdf.loc[len(stocksdf.index)] = [sector, symbol, 'Buy', sc]
            else: 
                stocksdf.loc[len(stocksdf.index)] = [sector, symbol, 'Sell', sc]

        stocksdf = stocksdf.sort_values(by=['Score'], ascending=False)
        buys = stocksdf[stocksdf['Technical Action'] == 'Buy']
        stocksdf.to_csv(self.path + '/results/' + sector + '-action.csv')
        stocksdf['date'] = datetime.now().strftime('%Y-%m-%d')
    
        # Connect to SQLite database (it will be created if it doesn't exist)
        conn = sqlite3.connect('data.db')
        
        # Save the DataFrame to the SQLite database
        stocksdf.to_sql('action', conn, if_exists='append', index=True)
        
        conn.close()


        buys.to_csv(self.path + '/results/' + sector + '-buys.csv')
        if (sector == 'QQQ' or sector =='SPY'):
            return d_list
        strongbuys = buys[buys['Score'] > 0]
        x = buys[buys['Score'] > 70]
        if x.shape[0] > 3:
            etfFraction = x.shape[0]
        d_list.append(strongbuys.head(etfFraction))

        return d_list



    def getPrices(self, ticker):
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

    def getETFaction(self, etf):
        prices = self.getPrices(etf)
        if prices['20DayEMA'].iloc[len(prices) - 1] > prices['100DayEMA'].iloc[len(prices) - 1]:
            return(3)
        else:
            return(1)

    def rsi(self, df, periods = 14, ema = True):
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


    def getScore(self, etf, stock, sharpe, columns):
        metricdf = pd.read_csv(self.path + '/metrics/' + etf + '-metrics.csv')
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        query = "SELECT * FROM metrics WHERE ETF = ?"
        metricdf = pd.read_sql_query(query, conn, params=(etf,))
                # Reversing the column name mapping
        metricdf = metricdf.rename(columns={
            'Dividend_Yield': 'Dividend Yield',
            'Forward_PE': 'Forward P/E',
            'Trailing_PE': 'Trailing P/E',
            'Market_Cap': 'Market Cap',
            'Trailing_EPS': 'Trailing EPS',
            'Forward_EPS': 'Forward EPS',
            'PEG_Ratio': 'PEG Ratio',
            'Price_To_Book': 'Price To Book',
            'EV_to_EBITDA': 'E/V to EBITDA',
            'Free_Cash_Flow': 'Free Cash Flow',
            'Debt_to_Equity': 'Debt to Equity',
            'Earnings_Growth': 'Earnings Growth',
            'Ebitda_Margins': 'Ebitda Margins',
            'Quick_Ratio': 'Quick Ratio',
            'Target_Mean_Price': 'Target Mean Price',
            'Return_on_Equity': 'Return on Equity',
            'Revenue_Growth': 'Revenue Growth',
            'Current_Ratio': 'Current Ratio',
            'Current_Price': 'Current Price'
        })

        conn.close()
        # metricdf = metricdf.fillna(0)
        factordict = {'Beta': -1 ,'Dividend Yield': 1, 'Forward P/E' : -1,'Trailing P/E': -1, 'Market Cap': 1, 'Trailing EPS': 1, 'Forward EPS': 1, 'PEG Ratio': -1, 'Price To Book': -1, 'E/V to EBITDA': -1, 'Free Cash Flow': 1, 'Debt to Equity': -1 ,'Earnings Growth': 1,'Ebitda margins': 1,'Quick Ratio': 1,'Target Mean Price': 1,'Return on Equity': 1 ,'Revenue Growth': 1,'Current Ratio': 1,'Current Price': 1}
        # print(metricdf.columns)
        # print(metricdf.columns[0])
        # subdf = metricdf[columns]
        #print(metricdf['Unnamed: 0'].head())
        score = 0
        # metricdf.rename(columns={metricdf.columns[0]:"Symbol"}, inplace=True)
        #print(metricdf['Symbol'].head())
        tickerrow = metricdf[metricdf['Ticker'] == stock]
        # print(stock)
        tickerrow = tickerrow.fillna(0)
        # print(tickerrow)
        # if etf == "XLV":
        #     print(stock)
        for x in columns.keys():
            if x == 'id' or x == 'ETF' or x == 'index' or x == 'date':
                continue
            mean = metricdf[x].mean()
            sd = metricdf[x].std()
            val = tickerrow[x].iloc[0]
            if val - mean > 0:
                val = val - mean
                val = val / sd
                val = val * factordict.get(x) * columns.get(x)
            else:
                val = mean - val
                val = val / sd
                val = val * factordict.get(x) * columns.get(x)
            # a = val / mean
            # if etf == "XLV":
            #     print(x)
            #     print(val)
                # print(a)
            score += val
        score += sharpe * 2
        return(score)
        
        
    def createGraphic(self, path):
        portfolio = pd.read_csv('{0}/portfolio/portfolio.csv'.format(path))
        filenames = self.loop(path,False)
        print(filenames)
        df2 = portfolio.groupby(['ETF'])['weight'].sum().reset_index()
        
        df2.sort_values(by='weight',inplace=True,ascending=False)
        df2 = df2.round(2)
        df2 = df2.reset_index()
        df2.to_csv('{0}/portfolio/sector_weights.csv'.format(path))

    def writePortfolioToLogs(self, path,portfolio):
        dat = date.today()
        dat = str(dat)
        filename = dat + '_portfolio.csv'
        portfolio.to_csv('{0}/logs/{1}'.format(path,filename))
        
        
    def findDifference(self, x,y):
        df1 = pd.read_csv(x)
        df2 = pd.read_csv(y)
        series1 = df1['Ticker']
        series2 = df2['Ticker']
        
        print(df2.head())
        union = pd.Series(np.union1d(series1, series2))
    
        # intersection of the series
        intersect = pd.Series(np.intersect1d(series1, series2))
        
        # uncommon elements in both the series 
        notcommonseries = union[~union.isin(intersect)]
        
        #print(notcommonseries)
        x = list(notcommonseries)
        
        #print(x)
        buys = df2.loc[df2['Ticker'].isin(x)]
        #print(buys.head())
        sells = df1.loc[df1['Ticker'].isin(x)]
        sells['Technical Action'] = 'Sell'
        final_df = pd.concat([buys,sells])
        final_df.to_csv('{0}/portfolio/actions.csv'.format(self.path),index=False)
    def getSentiment(self, stock):
        a = yf.Ticker(stock)
        from textblob import TextBlob
        # print(type(a.news))
        # print(a.news)
        if len(a.news) == 0:
            return 0
        score = 0
        for x in a.news:
            f = requests.get(x['link'])
            html_content = f.content

            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            # print(sentiment_pipeline(text))
            
            if sentiment > 0.2:
                score += 1
            else:
                score -= 1
        if score > 0:
            return 1
        else:
            return 0
    def getSentimentByNews(self, stock, tokenizer, model):
        
        a = yf.Ticker(stock)
        # Sample input text
        score = 0
        for x in a.news:
            f = requests.get(x['link'])
            html_content = f.content

            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text() 

        # Tokenize input
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

        # Make prediction
            outputs = model(**inputs)
            predictions = outputs.logits

        

            # Apply softmax to get probabilities
            probs = F.softmax(predictions, dim=1)

            # Get the predicted sentiment (0 for negative, 1 for positive)
            _, predicted_class = torch.max(probs, dim=1)
            predicted_sentiment = predicted_class.item()

            # Get the probability for the predicted sentiment
            predicted_prob = probs[0, predicted_class].item()
            
            if predicted_sentiment == 1 and predicted_prob > 0.65:
                score += 1
            elif predicted_sentiment != 1 and predicted_prob > 0.65:
                score -= 1
            else:
                continue
            
        if score > 0:
            return 1
        elif score == 0:
            return -1
        else: 
            return 0
            
    def getSentimentFast(self, stock):
        a = yf.Ticker(stock)
        sid = SentimentIntensityAnalyzer()
        score = 0
        for x in a.news:
            f = requests.get(x['link'])
            html_content = f.content

            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            sentiment_score = sid.polarity_scores(text)
            # print(sentiment_score)
            # Interpret the sentiment score
            if sentiment_score['compound'] >= 0.05:
                sentiment = 'Positive'
                score += 1
            elif sentiment_score['compound'] <= -0.05:
                sentiment = 'Negative'
                score -= 1
            else:
                sentiment = 'Neutral'

            # print(f"Sentiment: {sentiment}")
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    def calculateSentiments(self, etf, tokenizer, model):
        print('Processing: ' + etf)
        data = pd.read_csv('{0}/metrics/{1}-metrics.csv'.format(self.path, etf))
        data = data.rename(columns={'Unnamed: 0': 'Symbol'})
        values = []
        for x in data['Symbol']:
            val = self.getSentimentFast(x)
            values.append(val)
        data['Sentiment'] = val
        data.to_csv('{0}/metrics/{1}-metrics.csv'.format(self.path, etf),index=False)

    def __init__(self) -> None:
        self.path = os.getcwd()
        self.weightdict = {}
    
if __name__ == "__main__":
    cs = CalculateStocks()
    cs.main()