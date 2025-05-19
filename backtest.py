import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# === CONFIG ===
FMP_API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
START_DATE = datetime(2020, 5, 11)
END_DATE = datetime(2025, 5, 9)
REBALANCE_INTERVAL = timedelta(days=30)
TICKER = 'SPY'

# === Fetch full historical price data ===
def fetch_price_data(ticker):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?serietype=line&apikey={FMP_API_KEY}"
    r = requests.get(url)
    data = r.json().get("historical", [])
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[['date', 'close']].rename(columns={'close': 'Close'})
    return df

# === Backtest EMA crossover for SPY ===
def backtest_spy_compounding(df, starting_capital=5000):
    df['20EMA'] = df['Close'].rolling(window=20).mean()
    df['100EMA'] = df['Close'].rolling(window=100).mean()
    df['63d_momentum'] = df['Close'] / df['Close'].shift(63) - 1
    df = df.dropna().reset_index(drop=True)

    capital = starting_capital
    results = []
    current_date = START_DATE

    while current_date + REBALANCE_INTERVAL <= END_DATE:
        today_row = df[df['date'] <= current_date].iloc[-1]
        next_row = df[df['date'] <= current_date + REBALANCE_INTERVAL].iloc[-1]

        signal = 'Buy' if today_row['20EMA'] > today_row['100EMA'] else 'No Signal'
        ret = 0

        if signal == 'Buy':
            buy_price = today_row['Close']
            sell_price = next_row['Close']
            ret = (sell_price - buy_price) / buy_price
            capital *= (1 + ret)

        results.append({
            'Date': current_date.strftime("%Y-%m-%d"),
            'Signal': signal,
            'Return (%)': round(ret * 100, 2),
            'Capital ($)': round(capital, 2)
        })

        current_date += REBALANCE_INTERVAL

    return pd.DataFrame(results)


# === Run ===
if __name__ == "__main__":
    from yahooquery import Ticker

    def get_last_earnings_date_yahoo(ticker):
        t = Ticker(ticker)
        cal = t.earnings_calendar
        if cal and 'startdatetime' in cal[ticker][0]:
            return cal[ticker][0]['startdatetime'][:10]  # 'YYYY-MM-DD'
    print(get_last_earnings_date_yahoo("TSLA"))
    # df = fetch_price_data(TICKER)
    # result = backtest_spy_compounding(df)
    # result.to_csv("spy_ema_backtest.csv", index=False)
    # print(result)
