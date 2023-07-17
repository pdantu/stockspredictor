import yfinance as yf
import pandas as pd
from yahooquery import Ticker
import transformers
from transformers import pipeline
import requests
import urllib.request
from bs4 import BeautifulSoup
from yahooquery import Ticker
stock = 'AAPL'
ticker = Ticker(stock, asynchronous=True)
#print(ticker.summary_detail)
summary = ticker.summary_detail[stock]
    # print(summary)
default = ticker.key_stats[stock]
finance = ticker.financial_data[stock]
print(summary)
print(default)
# sentiment_pipeline =  pipeline('sentiment-analysis')
a = yf.Ticker('SPY')
from textblob import TextBlob
# print(type(a.news))
# print(a.news)
# print(a.news)
# for x in a.news:
#     f = requests.get(x['link'])
#     html_content = f.content

#     soup = BeautifulSoup(html_content, 'html.parser')
#     text = soup.get_text()
#     blob = TextBlob(text)
#     sentiment = blob.sentiment.polarity
#     # print(sentiment_pipeline(text))
#     print(sentiment)
#     score = 0
#     if sentiment > 0:
#         score += 1
#     else:
#         score -= 1
# print(score)