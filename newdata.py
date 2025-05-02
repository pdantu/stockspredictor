import os
import json
import pandas as pd
from datetime import datetime
import requests

# CONFIG
FMP_API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
path = os.getcwd()
sectorETF = ['XLK', 'XLF', 'XLU', 'XLI', 'XLE', 'XLV', 'XLP', 'XLY', 'XLC', 'XLRE', 'XLB']

# Load tickers from holdings CSVs
def preprocessAll(etflist):
    stocks = {}
    for etf in etflist:
        df = pd.read_csv(f'{path}/holdings/{etf}-holdings.csv')
        df = df[~df['Symbol'].isin(['SSIXX', 'Other', 'NLOK'])]
        stocks[etf] = df['Symbol'].tolist()

    with open("stocks.json", "w") as outfile:
        json.dump(stocks, outfile, indent=4)
    return stocks

# Get fundamental metrics from FMP
def get_fundamentals_fmp(ticker):
    base = "https://financialmodelingprep.com/api/v3"

    try:
        def safe_get_json(url, label):
            response = requests.get(url)
            if response.status_code != 200:
                print(f"⚠️  {label} error for {ticker}: HTTP {response.status_code}")
                return {}
            data = response.json()
            if not data:
                print(f"⚠️  {label} response empty for {ticker}")
                return {}
            return data[0]

        # Then use it:
        profile = safe_get_json(f"{base}/profile/{ticker}?apikey={FMP_API_KEY}", "profile")
        ratios = safe_get_json(f"{base}/ratios-ttm/{ticker}?apikey={FMP_API_KEY}", "ratios")
        metrics = safe_get_json(f"{base}/key-metrics-ttm/{ticker}?apikey={FMP_API_KEY}", "metrics")
        income_growth = safe_get_json(f"{base}/income-statement-growth/{ticker}?limit=1&apikey={FMP_API_KEY}", "income growth")
        analyst = safe_get_json(f"{base}/analyst-estimates/{ticker}?limit=1&apikey={FMP_API_KEY}", "analyst estimates")

        # profile = requests.get(f"{base}/profile/{ticker}?apikey={FMP_API_KEY}").json()[0]
        # ratios = requests.get(f"{base}/ratios-ttm/{ticker}?apikey={FMP_API_KEY}").json()[0]
        # metrics = requests.get(f"{base}/key-metrics-ttm/{ticker}?apikey={FMP_API_KEY}").json()[0]
    except Exception as e:
        print(f"Skipping {ticker} due to error: {e}")
        return None

    return {
        'Beta': profile.get('beta'),
        'Dividend_Yield': profile.get('lastDiv') / profile.get('price') if profile.get('lastDiv') and profile.get('price') else None,
        'Forward_PE': ratios.get('peRatioTTM'),
        'Trailing_PE': ratios.get('peRatioTTM'),
        'Market_Cap': profile.get('mktCap'),
        'Trailing_EPS': metrics.get('netIncomePerShareTTM'),
        'Forward_EPS': analyst.get('estimatedEpsAvg'),
        'PEG_Ratio': ratios.get('pegRatioTTM'),
        'Price_To_Book': ratios.get('priceToBookRatioTTM'),
        'EV_to_EBITDA': metrics.get('enterpriseValueOverEBITDATTM'),
        'Free_Cash_Flow': metrics.get('freeCashFlowPerShareTTM'),
        'Debt_to_Equity': ratios.get('debtEquityRatioTTM'),
        'Earnings_Growth': income_growth.get('growthNetIncome'),
        'Ebitda_Margins': income_growth.get('growthEBITDARatio'),
        'Quick_Ratio': ratios.get('quickRatioTTM'),
        'Target_Mean_Price': None,
        'Return_on_Equity': ratios.get('returnOnEquityTTM'),
        'Revenue_Growth': income_growth.get('growthRevenue'),
        'Current_Ratio': ratios.get('currentRatioTTM'),
        'Current_Price': profile.get('price')
    }

# Pull and save metrics for a sector ETF
def getSectData(sector, stock_list):
    output = []
    today = datetime.now().strftime('%Y-%m-%d')

    for symbol in stock_list:
        data = get_fundamentals_fmp(symbol)
        if data:
            data['ETF'] = sector
            data['Ticker'] = symbol
            data['date'] = today
            output.append(data)

    df = pd.DataFrame(output)
    os.makedirs(f'{path}/metrics', exist_ok=True)
    df.to_csv(f'{path}/metrics/{sector}-metrics.csv', index=False)
    print(f"Saved: metrics/{sector}-metrics.csv")

# Iterate over all sectors
def generateAll(stocks):
    for sector in stocks:
        print(f"Starting Sector: {sector}")
        getSectData(sector, stocks[sector])
        print(f"Finished Sector: {sector}")

# Entry point
def runAll():
    stocks = preprocessAll(sectorETF)
    print('finished preprocessing')
    generateAll(stocks)

def main():
    runAll()

if __name__ == "__main__":
    main()
