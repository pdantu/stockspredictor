import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# === Load FinBERT Sentiment Model ===
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

finbert = load_finbert()

# === Fetch earnings calendar and EPS surprise ===
def get_earnings_surprise(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v4/earning_calendar?symbol={ticker}&apikey={api_key}"
    today = datetime.today().date()
    two_weeks_ago = today - timedelta(days=30)

    try:
        response = requests.get(url)
        data = response.json()
        if not isinstance(data, list):
            return None

        for entry in data:
            if entry.get('symbol', '').upper() != ticker.upper():
                continue

            report_date = datetime.strptime(entry['date'], "%Y-%m-%d").date()
            if two_weeks_ago <= report_date <= today:
                eps_actual = entry.get('eps')
                eps_estimate = entry.get('epsEstimated')
                surprise = (eps_actual - eps_estimate) if eps_actual and eps_estimate else None
                return {
                    "ticker": ticker,
                    "report_date": report_date,
                    "eps_actual": eps_actual,
                    "eps_estimate": eps_estimate,
                    "surprise": surprise
                }
    except Exception as e:
        print(f"Error for {ticker}: {e}")
    return None


# === Price Fetching and Momentum ===
def get_prices(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?serietype=line&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json().get("historical", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df[['date', 'close']].rename(columns={'close': 'Close'})
        return df
    except Exception as e:
        print(f"⚠️ Error fetching price for {ticker}: {e}")
        return pd.DataFrame()

def get_momentum_after_earnings(ticker, report_date, api_key, days_after=3):
    df = get_prices(ticker, api_key)
    if df.empty:
        return None
    df = df.set_index('date')
    try:
        start_price = df.loc[report_date]['Close']
        target_date = report_date + timedelta(days=days_after)
        valid_dates = df.index[df.index >= target_date]
        if not valid_dates.empty:
            end_price = df.loc[valid_dates[0]]['Close']
            return (end_price - start_price) / start_price
    except Exception as e:
        print(f"⚠️ Momentum error for {ticker}: {e}")
    return None

# === Sentiment Score Using FinBERT ===
def get_stock_news(ticker, limit, api_key):
    url = f"https://financialmodelingprep.com/api/v3/stock_news"
    params = {
        "tickers": ticker,
        "limit": limit,
        "apikey": api_key
    }
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return []

def sentiment_during_earnings(ticker, report_date, api_key):
    news_items = get_stock_news(ticker, limit=10, api_key=api_key)
    window_start = report_date
    window_end = report_date + timedelta(days=3)

    scores = []
    for item in news_items:
        try:
            pub_date = datetime.strptime(item['publishedDate'], "%Y-%m-%d %H:%M:%S").date()
            if window_start <= pub_date <= window_end:
                snippet = f"{item['title']}. {item.get('text', '')[:1000]}"
                result = finbert(snippet)[0]
                if result['label'].lower() == 'positive':
                    scores.append(result['score'])
        except:
            continue

    return np.mean(scores) if scores else 0

# === Final Stock Screener ===
def select_candidates(tickers, api_key):
    candidates = []

    for ticker in tickers:
        print(f"\nChecking {ticker}...")

        er = get_earnings_surprise(ticker, api_key)
        if not er:
            print(f"Skipped {ticker}: No recent earnings report found.")
            continue
        if er['surprise'] is None:
            print(f"Skipped {ticker}: EPS or estimate missing.")
            continue
        if er['surprise'] < 0:
            print(f"Skipped {ticker}: Negative earnings surprise ({er['surprise']:.2f}).")
            continue
        print(f"EPS Surprise: {er['surprise']:.2f}")

        momentum = get_momentum_after_earnings(ticker, er['report_date'], api_key)
        if momentum is None:
            print(f"Skipped {ticker}: Could not compute momentum.")
            continue
        if momentum < 0.02 or momentum > 0.1:
            print(f"Skipped {ticker}: Momentum out of range ({momentum:.2%}).")
            continue
        print(f"Momentum: {momentum:.2%}")

        sentiment = sentiment_during_earnings(ticker, er['report_date'], api_key)
        if sentiment < 0.9:
            print(f"Skipped {ticker}: Sentiment score too low ({sentiment:.2f}).")
            continue
        print(f"Sentiment Score: {sentiment:.2f}")

        candidates.append({
            "Ticker": ticker,
            "Report Date": er['report_date'],
            "EPS Surprise": er['surprise'],
            "Momentum": round(momentum, 4),
            "Sentiment Score": round(sentiment, 4)
        })

    return pd.DataFrame(candidates)


# === Run the Strategy ===
if __name__ == "__main__":
    API_KEY = "uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB"
    SAMPLE_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'WMT', 'TSLA', 'META', 'NFLX', 'CRM', 'GOOG', 'AMD', 'NVDA', 'INTC', 'DIS', 'BA', 'VZ', 'KO', 'PEP', 'JNJ', 'PFE']

    print("🔍 Running Earnings + Momentum Drift Screener...")
    final_df = select_candidates(SAMPLE_TICKERS, API_KEY)

    if not final_df.empty:
        print("\nSelected Stocks:\n")
        print(final_df)
        final_df.to_csv("post_earnings_drift_candidates.csv", index=False)
    else:
        print("No qualifying candidates found.")
