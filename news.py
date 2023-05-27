import yfinance as yf
import pandas as pd
from yahooquery import Ticker
import transformers
from transformers import pipeline
import requests
import urllib.request
from bs4 import BeautifulSoup
# sentiment_pipeline =  pipeline('sentiment-analysis')
a = yf.Ticker('MSFT')
from textblob import TextBlob
# print(type(a.news))
# print(a.news)

for x in a.news:
    f = requests.get(x['link'])
    html_content = f.content

    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    # print(sentiment_pipeline(text))
    print(sentiment)
    