import requests
from datetime import datetime, timedelta
from transformers import pipeline
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load once at the top
finbert = load_finbert()
import requests
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta

def has_recent_or_upcoming_earnings(symbol, api_key):
    url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}?apikey={api_key}"
    today = datetime.today().date()
    two_weeks_ago = today - timedelta(days=14)
    two_weeks_ahead = today + timedelta(days=14)

    try:
        response = requests.get(url)
        data = response.json()
    except:
        return False

    if not isinstance(data, list):
        return False

    for entry in data:
        try:
            report_date = datetime.strptime(entry['date'], "%Y-%m-%d").date()
            if two_weeks_ago <= report_date <= two_weeks_ahead:
                return True
        except:
            continue

    return False


# API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
# symbol = "WMT"
# from_date = "2023-08-01"
# to_date = "2023-10-31"
# print(f"Symbol: {symbol} ({type(symbol)})")

# symbol = "WMT"
# url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}?apikey={API_KEY}"
# url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}?apikey={API_KEY}"
# today = datetime.today().date()
# two_weeks_ago = today - timedelta(days=14)

# response = requests.get(url)
# data = response.json()

# recent_reports = []
# if isinstance(data, list):
#     print(list(data[0].keys()))
    
#     for entry in data:
#         try:
#             report_date = datetime.strptime(entry['date'], "%Y-%m-%d").date()
#             if two_weeks_ago <= report_date <= today:
#                 print(f"{symbol} reported on {report_date}, EPS: {entry.get('eps')}, Estimate: {entry.get('epsEstimated')}")
#                 recent_reports.append(entry)
#         except:
#             continue

# Load sentiment model
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt",  # Force PyTorch
    device=-1         # CPU
)
# Fetch news for a specific ticker
def get_stock_news(ticker, limit, api_key):
    url = f"https://financialmodelingprep.com/api/v3/stock_news"
    params = {
        "tickers": ticker,
        "limit": limit,
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    return response.json()

# Run sentiment on each article
def get_positive_news_for_stock(ticker, api_key, limit=10, threshold=0.95):
    news_items = get_stock_news(ticker, limit, api_key)
    print(f"\n🔍 Found {len(news_items)} news items for {ticker}")

    positive_news = []

    for i, item in enumerate(news_items):
        headline = item["title"]
        text = item.get("text", "")
        snippet = f"{headline}. {text}"

        print(f"\n📰 News {i+1}: {headline}")
        try:
            result = sentiment_model(snippet)[0]
            print(f"  → Sentiment: {result['label']} (score: {result['score']:.3f})")

            if result["label"] == "POSITIVE" and result["score"] >= threshold:
                positive_news.append({
                    "date": item["publishedDate"],
                    "headline": headline,
                    "score": result["score"],
                    "source": item["publisher"],
                    "url": item["url"]
                })
        except Exception as e:
            print(f"  ⚠️ Error analyzing sentiment: {e}")
            continue

    print(f"\n✅ Found {len(positive_news)} very positive articles for {ticker}")
    df = pd.DataFrame(positive_news)
    return df

def get_sentiment_score_simple(ticker, api_key):
    news_items = get_stock_news(ticker, limit=10, api_key=api_key)

    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    label_scores = []

    for item in news_items:
        snippet = f"{item['title']}. {item.get('text', '')[:1000]}"
        try:
            result = finbert(snippet)[0]
            label = result["label"].lower()
            label_scores.append(label_map.get(label, 0))
        except:
            continue

    final_score = sum(label_scores) / len(label_scores) if label_scores else 0
    print(f"Final sentiment score for {ticker}: {final_score:.3f}")
    return final_score

def get_positive_news_for_stock_finbert(ticker, api_key, limit=10, threshold=0.95):
    news_items = get_stock_news(ticker, limit, api_key)
    print(f"\n🔍 Found {len(news_items)} news items for {ticker}")

    positive_news = []

    for i, item in enumerate(news_items):
        headline = item["title"]
        text = item.get("text", "")
        snippet = f"{headline}. {text[:1000]}"

        print(f"\n📰 News {i+1}: {headline}")
        print(f"  → Preview: {snippet[:150]}")

        try:
            result = finbert(snippet)[0]
            print(f"  → FinBERT Sentiment: {result['label']} (score: {result['score']:.3f})")

            if result["label"].lower() == "positive" and result["score"] >= 0.8:
                positive_news.append({
                    "date": item["publishedDate"],
                    "headline": headline,
                    "score": result["score"],
                    # "source": item["publisher"],
                    "url": item["url"]
                })

        except Exception as e:
            print(f"  ⚠️ Error analyzing sentiment: {e}")
            continue

    print(f"\n✅ FinBERT found {len(positive_news)} very positive articles for {ticker}")
    return pd.DataFrame(positive_news)

import os
import pandas as pd

SECTOR_ETFS = {'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU'}

def get_sentiment_score_scaled(ticker, api_key):
    news_items = get_stock_news(ticker, limit=10, api_key=api_key)

    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    label_scores = []

    for item in news_items:
        snippet = f"{item['title']}. {item.get('text', '')[:1000]}"
        try:
            result = finbert(snippet)[0]
            label = result["label"].lower()
            label_scores.append(label_map.get(label, 0))
        except:
            continue

    if label_scores:
        raw_score = sum(label_scores) / len(label_scores)
        scaled_score = int(raw_score * 100)
    else:
        scaled_score = 0

    print(f"📊 {ticker} sentiment score: {scaled_score}")
    return scaled_score

def process_sector_etf_holdings(api_key, holdings_dir="holdings"):
    results = []

    for file in os.listdir(holdings_dir):
        if not file.endswith("-holdings.csv"):
            continue

        etf_name = file.replace("-holdings.csv", "")
        if etf_name not in SECTOR_ETFS:
            continue

        path = os.path.join(holdings_dir, file)
        df = pd.read_csv(path)
        print(df.columns)
        if "Symbol" not in df.columns:
            continue

        tickers = df['Symbol'].dropna().unique().tolist()

        for ticker in tickers:
            try:
                score = get_sentiment_score_scaled(ticker, api_key)
                results.append({
                    "ETF": etf_name,
                    "Ticker": ticker,
                    "SentimentScore": score
                })
            except Exception as e:
                print(f"⚠️ Failed on {ticker} in {etf_name}: {e}")
                continue

    return pd.DataFrame(results)

# === Run It ===
if __name__ == "__main__":
    api_key = "uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB"
    sentiment_df = process_sector_etf_holdings(api_key, holdings_dir="holdings")
    sentiment_df.to_csv("sentiment_scores.csv", index=False)
    print("\n✅ Saved sentiment_scores.csv")
