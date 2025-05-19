import os
import pandas as pd
import requests
from datetime import datetime
from typing import Literal

# CONFIG
API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
RESULTS_DIR = 'results'
OUTPUT_PATH = 'portfolio/portfolio_combined.csv'

def get_macro_signal() -> Literal["Growth", "Value", "Income", "Balanced"]:
    try:
        def safe_last_value(data, label, fallback):
            if isinstance(data, list) and len(data) > 0 and 'value' in data[-1]:
                return data[-1]['value']
            print(f"⚠️ No valid data returned for {label}, using fallback: {fallback}")
            return fallback

        inflation = requests.get(f'https://financialmodelingprep.com/api/v4/inflation?apikey={API_KEY}').json()
        treasury = requests.get(f'https://financialmodelingprep.com/api/v4/treasury?apikey={API_KEY}').json()
        gdp = requests.get(f'https://financialmodelingprep.com/api/v4/usgdp?apikey={API_KEY}').json()
        unemp = requests.get(f'https://financialmodelingprep.com/api/v4/unemployment?apikey={API_KEY}').json()

        inflation_rate = safe_last_value(inflation, "inflation", 2.5)
        treasury_yield = safe_last_value(treasury, "treasury", 3.0)
        gdp_growth = safe_last_value(gdp, "gdp", 1.5)
        unemployment_rate = safe_last_value(unemp, "unemployment", 4.0)

        if inflation_rate > 3.5 and treasury_yield > 4:
            return "Value"
        elif gdp_growth > 2.5 and unemployment_rate < 4:
            return "Growth"
        elif treasury_yield < 2 and inflation_rate < 2:
            return "Income"
        else:
            return "Balanced"

    except Exception as e:
        print(f"🔥 Error fetching macro data: {e}")
        return "Balanced"

def merge_portfolios(macro_signal: str):
    weights = {
        "Growth": {'growth': 0.6, 'value': 0.2, 'income': 0.2},
        "Value": {'growth': 0.2, 'value': 0.6, 'income': 0.2},
        "Income": {'growth': 0.2, 'value': 0.2, 'income': 0.6},
        "Balanced": {'growth': 1/3, 'value': 1/3, 'income': 1/3}
    }

    combined = []
    for style, weight in weights[macro_signal].items():
        file_path = os.path.join(RESULTS_DIR, f'portfolio{style}.csv')
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df = df.copy()
        df['style'] = style

        # Take top N stocks proportionally (e.g. total ~15 stocks)
        top_n = max(1, int(weight * 15))
        df_top = df.nlargest(top_n, 'Score')
        df_top['weight'] = (df_top['Score'] / df_top['Score'].sum()) * weight * 100
        combined.append(df_top)

    if combined:
        final = pd.concat(combined, ignore_index=True)
        final = final.sort_values(by='weight', ascending=False)
        os.makedirs('portfolio', exist_ok=True)
        final.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Combined portfolio saved: {OUTPUT_PATH}")
    else:
        print("⚠️ No portfolios found to merge.")

if __name__ == '__main__':
    signal = get_macro_signal()
    print(f"Macro Style Selected: {signal}")
    merge_portfolios(signal)
