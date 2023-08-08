import yfinance as yf
import pandas as pd
from yahooquery import Ticker
import transformers
from transformers import pipeline
import requests
import urllib.request
from bs4 import BeautifulSoup
from yahooquery import Ticker
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# stock = 'AAPL'
# ticker = Ticker(stock, asynchronous=True)
# #print(ticker.summary_detail)
# summary = ticker.summary_detail[stock]
#     # print(summary)
# default = ticker.key_stats[stock]
# finance = ticker.financial_data[stock]
# print(summary)
# print(default)
# sentiment_pipeline =  pipeline('sentiment-analysis')
a = yf.Ticker('EFX')

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Sample input text
score = 0
for x in a.news:
    f = requests.get(x['link'])
    # print(x['link'])
    html_content = f.content

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')

    # Extract the Text
    text = "\n".join([p.get_text() for p in paragraphs])

    # text = soup.get_text() 
    
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

# Print the results
    print("Predicted Sentiment:", "Positive" if predicted_sentiment == 1 else "Negative")
    print("Probability:", predicted_prob)
    print(predicted_sentiment)

    if predicted_sentiment == 1 and predicted_prob > 0.65:
        score += 1
    elif predicted_sentiment != 1 and predicted_prob > 0.65:
        print('hi')
        score -= 1
    else:
        continue


if score > 0:
    print(1)
elif score == 0:
    print(0)
else:
    print(-1)




from textblob import TextBlob

score = 0
for x in a.news:
    f = requests.get(x['link'])
    html_content = f.content

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')

    # Extract the Text
    text = "\n".join([p.get_text() for p in paragraphs])
    # text = soup.get_text()
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    # print(sentiment_pipeline(text))
    print(sentiment)
    
    if sentiment > 0.2:
        score += 1
    else:
        score -= 1
print(score)

def preprocess_text(text):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove punctuation and non-alphabetic characters
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(filtered_tokens)

sid = SentimentIntensityAnalyzer()
score = 0
for x in a.news:
    f = requests.get(x['link'])
    html_content = f.content

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')

    # Extract the Text
    text = "\n".join([p.get_text() for p in paragraphs])
    # text = soup.get_text()
    preprocessed_text = preprocess_text(text)
    sentiment_score = sid.polarity_scores(preprocessed_text)
    print(sentiment_score)
    # Interpret the sentiment score
    if sentiment_score['compound'] >= 0.05:
        sentiment = 'Positive'
        score += 1
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
        score -= 1
    else:
        sentiment = 'Neutral'

    print(f"Sentiment: {sentiment}")
print(score)