import requests
import pandas as pd
import matplotlib.pyplot as plt

FMP_API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
ticker = 'NLSN'

# 1. Fetch fundamentals
base = "https://financialmodelingprep.com/api/v3"
profile_url = f"{base}/profile/{ticker}?apikey={FMP_API_KEY}"
ratios_url = f"{base}/ratios-ttm/{ticker}?apikey={FMP_API_KEY}"
metrics_url = f"{base}/key-metrics-ttm/{ticker}?apikey={FMP_API_KEY}"

try:
    profile = requests.get(profile_url).json()[0]
    ratios = requests.get(ratios_url).json()[0]
    metrics = requests.get(metrics_url).json()[0]

    print("=== Company Profile ===")
    print(f"Name: {profile.get('companyName')}")
    print(f"Industry: {profile.get('industry')}")
    print(f"Market Cap: {profile.get('mktCap')}")
    print(f"Price: {profile.get('price')}")
    print()

    print("=== Key Ratios ===")
    for key in ['peRatioTTM', 'pegRatioTTM', 'returnOnEquityTTM', 'debtEquityRatioTTM']:
        print(f"{key}: {ratios.get(key)}")

    print()

    print("=== Key Metrics ===")
    for key in ['netIncomePerShareTTM', 'freeCashFlowPerShareTTM', 'enterpriseValueOverEBITDATTM']:
        print(f"{key}: {metrics.get(key)}")

except Exception as e:
    print(f"❌ Failed to fetch fundamentals: {e}")

# 2. Fetch and plot price history
price_url = f"{base}/historical-price-full/{ticker}?serietype=line&apikey={FMP_API_KEY}"
resp = requests.get(price_url)
price_data = resp.json().get("historical", [])

if not price_data:
    print(f"⚠️ No price data found for {ticker}")
else:
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    plt.figure(figsize=(10, 4))
    plt.plot(df['date'], df['close'], label='Close Price')
    plt.title(f"{ticker} Price History")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
