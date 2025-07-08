import requests
import time

# ========== CONFIG ==========
FMP_API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
TICKERS = ['AAPL', 'TSLA', 'NVDA']
TICKERS = ['FLR', 'CRCL']
LLAMA_URL = "http://localhost:11434/api/chat"
# ============================

def llama3(prompt):
    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(LLAMA_URL, headers=headers, json=data)
    return response.json()["message"]["content"]

def analyze_stock_news(news_text, ticker):
    prompt = f"""
You are a financial analyst AI. Given the news below about {ticker}, decide:

Will this news likely cause a **positive shift** in the stock price in the next trading session? Answer "YES" or "NO" and provide a short reasoning (1-2 sentences).

News: {news_text}
"""
    return llama3(prompt)

def get_stock_news(ticker, limit, api_key):
    url = f"https://financialmodelingprep.com/api/v3/stock_news"
    params = {
        "tickers": ticker,
        "limit": limit,
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    print(f"{ticker} Status Code: {response.status_code}")
    print(f"{ticker} Response Text: {response.text}")
    return response.json()

def main():
    seen_headlines = set()
    print("🔎 Starting FMP News + LLaMA analysis (single run)...")
    try:
        for ticker in TICKERS:
            news_items = get_stock_news(ticker, limit=3, api_key=FMP_API_KEY)
            for item in news_items:
                headline = item['title']

                if headline not in seen_headlines:
                    seen_headlines.add(headline)
                    print(f"\n📢 New headline for {ticker}: {headline}")

                    result = analyze_stock_news(headline, ticker)
                    print(f"🧠 LLaMA Analysis: {result}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
