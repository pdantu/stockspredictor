import pandas as pd
import requests

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
    df = pd.read_csv('portfolio/portfoliogrowth.csv')
    df['Current Price'] = df['Ticker'].apply(get_current_price)
    df['Shares'] = df['Dollar Amount'] / df['Current Price']
    df['Shares'] = df['Shares'].round(2)

    df.to_csv('portfolio/portfoliogrowth_with_shares.csv', index=False)
    print('Saved: portfolio/portfoliogrowth_with_shares.csv')

if __name__ == "__main__":
    main()
