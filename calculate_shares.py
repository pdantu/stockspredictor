import pandas as pd
import requests
import os

FMP_API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'

def get_current_price(ticker):
    url = f'https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}'
    try:
        response = requests.get(url)
        data = response.json()
        return data[0]['price'] if data else None
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return None

def main():
    portfolio_types = ['value', 'income']
    path = os.getcwd()

    for ptype in portfolio_types:
        file_path = f'{path}/portfolio/portfolio{ptype}.csv'
        df = pd.read_csv(file_path)
        print(f'Processing: {file_path}')
        print(df['Ticker'])

        df['Current Price'] = df['Ticker'].apply(get_current_price)
        df['Shares'] = df['Dollar Amount'] / df['Current Price']
        df['Shares'] = df['Shares'].round(2)
        df['mainShares'] = df['Shares'] * 0.4

        output_path = f'{path}/portfolio/portfolio{ptype}_with_shares.csv'
        df.to_csv(output_path, index=False)
        print(f'Saved: {output_path}')

if __name__ == "__main__":
    main()
