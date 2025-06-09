import requests

FMP_API_KEY = 'uv0qboZyS4HJzfriyBUf9Q9RZJDgw0pB'
symbol = 'MCHP'

url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={FMP_API_KEY}"

try:
    response = requests.get(url)
    data = response.json()
    if isinstance(data, list) and data:
        trailing_pe = data[0].get('peRatioTTM')
        print(f"Trailing P/E for {symbol}: {trailing_pe}")
    else:
        print(f"No data returned for {symbol}")
except Exception as e:
    print(f"Error fetching Trailing P/E: {e}")
